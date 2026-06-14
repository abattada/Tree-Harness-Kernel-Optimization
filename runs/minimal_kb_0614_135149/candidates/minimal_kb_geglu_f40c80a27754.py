import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def geglu_kernel(
    x_ptr,             # pointer to input [M, K] (flattened)
    out_ptr,           # pointer to output [M, N]
    stride_x_row,      # row stride of x in elements (K)
    stride_out_row,    # row stride of out (N)
    BLOCK_SIZE: tl.constexpr,   # must equal N (4096)
    N: tl.constexpr,            # output width = K//2
):
    # Safety check: the block covers a full output row → no masking needed
    tl.static_assert(BLOCK_SIZE == N, "BLOCK_SIZE must equal N")

    # Row index
    row = tl.program_id(0)

    # Column offsets (full row, always valid)
    col_offs = tl.arange(0, BLOCK_SIZE)

    # Base pointers for this row
    x_row  = x_ptr  + row * stride_x_row
    out_row = out_ptr + row * stride_out_row

    # Load a (first half) and b (second half) – streaming data, evict after use
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
    x: float32 tensor of shape [8192, 8192]
    returns: float32 tensor of shape [8192, 4096]
    """
    M, K = x.shape
    N = K // 2                     # 4096

    # Allocate output tensor
    out = torch.empty(M, N, device=x.device, dtype=torch.float32)

    BLOCK_SIZE = N                 # covers full output row

    # One program per row
    grid = (M,)

    geglu_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        N=N,
        num_warps=16,              # increased warps to 16 → each thread handles 8 elements (down from 16)
        num_stages=2,              # fewer stages to balance register/smem with higher warp count
    )

    return out