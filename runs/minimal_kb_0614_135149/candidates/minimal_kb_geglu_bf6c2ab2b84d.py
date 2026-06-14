import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def geglu_kernel(
    x_ptr,               # input tensor [M, K] flattened
    out_ptr,             # output tensor [M, N] flattened
    stride_x_row,        # row stride of x in elements (K)
    stride_out_row,      # row stride of out in elements (N)
    M: tl.constexpr,     # total number of rows (8192)
    N: tl.constexpr,     # output width = K//2 (4096)
    BLOCK_SIZE: tl.constexpr,  # must equal N
    GRID_SIZE: tl.constexpr,   # number of persistent programs to launch
):
    # Safety: the block covers a full output row → no masks needed
    tl.static_assert(BLOCK_SIZE == N, "BLOCK_SIZE must equal N")

    pid = tl.program_id(0)

    # Grid‑stride loop: each program processes multiple rows
    for row in range(pid, M, GRID_SIZE):
        # Base pointers for this row
        x_row   = x_ptr   + row * stride_x_row
        out_row = out_ptr + row * stride_out_row

        # Column offsets (full row, always in bounds)
        col_offs = tl.arange(0, BLOCK_SIZE)

        # Load a (first half) and b (second half) – streaming, evict after use
        a = tl.load(x_row + col_offs,     eviction_policy='evict_first')
        b = tl.load(x_row + col_offs + N, eviction_policy='evict_first')

        # GELU with tanh approximation:
        #   GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        sqrt2_over_pi = 0.7978845608028654
        c = 0.044715

        a3    = a * a * a
        inner = sqrt2_over_pi * (a + c * a3)
        tanh_val = libdevice.tanh(inner)
        gelu_a = 0.5 * a * (1.0 + tanh_val)

        # Gating
        out = gelu_a * b

        # Store result (streaming write)
        tl.store(out_row + col_offs, out, eviction_policy='evict_first')


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    x: float32 tensor of shape [M, K] with K even
    returns: float32 tensor of shape [M, K//2]
    """
    M, K = x.shape
    N = K // 2                     # 4096

    # Allocate output
    out = torch.empty(M, N, device=x.device, dtype=torch.float32)

    BLOCK_SIZE = N                 # covers whole output row

    # Launch a small number of persistent programs (one per row is overkill)
    GRID_SIZE = 256 if M >= 256 else M
    grid = (GRID_SIZE,)

    geglu_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        M=M, N=N,
        BLOCK_SIZE=BLOCK_SIZE,
        GRID_SIZE=GRID_SIZE,
        num_warps=8,              # balanced register/occupancy
        num_stages=4,             # moderate software pipelining
    )

    return out