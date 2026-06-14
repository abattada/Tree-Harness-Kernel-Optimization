import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ---------------------------------------------------------------------------
# GeGLU kernel (gated GELU with tanh approximation)
# Grid-stride loop over rows to amortise kernel launch overhead.
# One block per row, BLOCK_SIZE == N (full output row) -> no masks needed.
# ---------------------------------------------------------------------------

@triton.jit
def geglu_kernel(
    x_ptr,                     # input tensor (M, K) flattened
    out_ptr,                   # output tensor (M, N) flattened, N = K//2
    stride_x_row,              # row stride of x in elements (= K)
    stride_out_row,            # row stride of out in elements (= N)
    M,                         # number of rows (runtime)
    N: tl.constexpr,           # output dimension = K//2
    BLOCK_SIZE: tl.constexpr,  # must equal N for this specialisation
):
    tl.static_assert(BLOCK_SIZE == N, "BLOCK_SIZE must equal N")

    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)  # dynamic grid size

    # Process rows in a grid-stride loop – kills launch overhead
    for row in range(pid, M, num_programs):
        # Column indices (exactly the full row, no masking)
        col_offs = tl.arange(0, BLOCK_SIZE)

        # Pointers to the current row
        x_row = x_ptr + row * stride_x_row
        out_row = out_ptr + row * stride_out_row

        # Load first half (a) and second half (b)
        a = tl.load(x_row + col_offs,        eviction_policy='evict_first')
        b = tl.load(x_row + col_offs + N,    eviction_policy='evict_first')

        # GELU(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
        sqrt2_over_pi = 0.7978845608028654
        c = 0.044715

        a3 = a * a * a
        inner = sqrt2_over_pi * (a + c * a3)
        tanh_val = libdevice.tanh(inner)
        gelu_a = 0.5 * a * (1.0 + tanh_val)

        # Gating
        out = gelu_a * b

        # Store result
        tl.store(out_row + col_offs, out, eviction_policy='evict_first')


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    x: float32 tensor of shape [M, K] with K even, here [8192, 8192].
    returns: float32 tensor of shape [M, K//2] = [8192, 4096].
    """
    M, K = x.shape
    N = K // 2                     # 4096

    out = torch.empty(M, N, device=x.device, dtype=torch.float32)

    BLOCK_SIZE = N                 # full row in one block
    # Launch many fewer programs than rows to radically cut launch overhead
    grid_size = min(M, 1024)       # 1024 programs, each doing 8 rows on average
    grid = (grid_size,)

    geglu_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        M, N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,               # good occupancy/register balance
        num_stages=4,              # moderate pipelining
    )

    return out