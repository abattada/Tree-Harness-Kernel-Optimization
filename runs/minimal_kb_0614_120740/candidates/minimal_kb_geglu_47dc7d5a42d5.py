import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel: GeGLU (gated GELU with tanh approximation)
# Input:  x [M, K] with K even, e.g., (8192, 8192)
# Output: out [M, K//2] = (8192, 4096)
#
# One program per row, BLOCK_SIZE == N (output dimension). No masks.
# Uses the identity tanh(x) = 2*sigmoid(2*x)-1 with tl.exp2 to replace the
# libdevice.tanh call for potentially higher throughput.
# ---------------------------------------------------------------------------

@triton.jit
def geglu_kernel(
    x_ptr,              # pointer to input (2D)
    out_ptr,            # pointer to output (2D)
    stride_x_row,       # row stride of x (in elements)
    stride_out_row,     # row stride of out (in elements)
    BLOCK_SIZE: tl.constexpr,   # = N
    N: tl.constexpr,            # output dimension (same as BLOCK_SIZE)
):
    # Static assertion: full row, no masking needed
    tl.static_assert(BLOCK_SIZE == N)

    # Row index
    row = tl.program_id(0)

    # Column offsets (0..BLOCK_SIZE-1)
    col_offs = tl.arange(0, BLOCK_SIZE)

    # Pointers to this row
    x_row = x_ptr + row * stride_x_row
    out_row = out_ptr + row * stride_out_row

    # Load a (first half) and b (second half) – evict_first
    a = tl.load(x_row + col_offs, eviction_policy='evict_first')
    b = tl.load(x_row + col_offs + N, eviction_policy='evict_first')

    # ---- GELU_tanh(a) ---------------------------------------------------
    #  GELU(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
    sqrt2_over_pi = 0.7978845608028654   # sqrt(2/pi)
    c = 0.044715

    a3 = a * a * a
    inner = sqrt2_over_pi * (a + c * a3)

    # tanh(inner) = 2 * sigmoid(2*inner) - 1
    # sigmoid(x) = 1 / (1 + exp(-x))
    # exp(-x) = exp2(-x * log2(e))
    log2e = 1.4426950408889634
    two = 2.0
    one = 1.0

    inner2 = two * inner
    exp_arg = -inner2 * log2e
    exp_val = tl.exp2(exp_arg)           # exp(-2*inner)
    sigmoid = one / (one + exp_val)
    tanh_val = two * sigmoid - one

    gelu_a = 0.5 * a * (one + tanh_val)

    # ---- Gate -----------------------------------------------------------
    out = gelu_a * b

    # ---- Store result ---------------------------------------------------
    tl.store(out_row + col_offs, out, eviction_policy='evict_first')


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    x: float32 tensor of shape [M, K] (K even)
    Returns: float32 tensor of shape [M, K//2]
    """
    M, K = x.shape
    N = K // 2  # output dimension

    out = torch.empty(M, N, device=x.device, dtype=torch.float32)

    BLOCK_SIZE = 4096  # equals N
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