import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# GeGLU kernel – free refinement round
# Primary change: use tl.sigmoid instead of libdevice.tanh for the tanh
# computation inside GELU, which can be slightly faster while remaining
# mathematically equivalent.
#
# Input:  x [8192, 8192]
# Output: out [8192, 4096]
# ---------------------------------------------------------------------------

@triton.jit
def geglu_kernel(
    x_ptr,                # pointer to input (2D flat)
    out_ptr,              # pointer to output (2D flat)
    stride_x,             # row stride of x in elements (= 8192)
    stride_out,           # row stride of out in elements (= 4096)
    BLOCK_SIZE: tl.constexpr,   # must equal N
    N: tl.constexpr,            # output dimension = K//2 (= 4096)
):
    # Full-row coverage: no masks needed
    tl.static_assert(BLOCK_SIZE == N,
                     "BLOCK_SIZE must equal N for this specialization")

    row = tl.program_id(0)
    col_offs = tl.arange(0, BLOCK_SIZE)

    x_row = x_ptr + row * stride_x
    out_row = out_ptr + row * stride_out

    # Load a (first half) and b (second half) – read once, evict afterwards
    a = tl.load(x_row + col_offs,           eviction_policy='evict_first')
    b = tl.load(x_row + col_offs + N,       eviction_policy='evict_first')

    # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt2_over_pi = 0.7978845608028654
    c = 0.044715

    a3 = a * a * a
    inner = sqrt2_over_pi * (a + c * a3)

    # Replace libdevice.tanh with the equivalent 2*sigmoid(2*x)-1
    tanh_val = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
    gelu_a = 0.5 * a * (1.0 + tanh_val)

    out = gelu_a * b

    tl.store(out_row + col_offs, out, eviction_policy='evict_first')


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    x: float32 tensor of shape [8192, 8192]
    returns: float32 tensor of shape [8192, 4096]
    """
    M, K = x.shape
    N = K // 2                          # 4096

    out = torch.empty(M, N, device=x.device, dtype=x.dtype)

    BLOCK_SIZE = N
    grid = (M,)

    geglu_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        N=N,
        num_warps=8,
        num_stages=4,
    )

    return out