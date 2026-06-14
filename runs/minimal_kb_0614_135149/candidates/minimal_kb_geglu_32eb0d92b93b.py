import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel: GeGLU (gated GELU with tanh approximation)
# Input:  x [M, K]  with K even (here [8192, 8192])
# Output: out [M, K//2] = [8192, 4096]
# One program per row, BLOCK_SIZE equals N for full-row processing.
# The core improvement: replace tanh with sigmoid via the identity
#   0.5*x*(1+tanh(y)) = x*sigmoid(2y)
# which reduces arithmetic and uses the faster tl.sigmoid builtin.
# ---------------------------------------------------------------------------

@triton.jit
def geglu_kernel(
    x_ptr,
    out_ptr,
    stride_x,
    stride_out,
    BLOCK_SIZE: tl.constexpr,
):
    # Row index
    row = tl.program_id(0)

    # Column offsets for the full output row (no mask needed)
    col_offs = tl.arange(0, BLOCK_SIZE)

    # Row pointers
    x_row = x_ptr + row * stride_x
    out_row = out_ptr + row * stride_out

    # Load a (first half) and b (second half)
    a = tl.load(x_row + col_offs, eviction_policy='evict_first')
    b = tl.load(x_row + col_offs + BLOCK_SIZE, eviction_policy='evict_first')

    # GELU with tanh approximation:
    #   gelu(x) = x * sigmoid(2 * sqrt(2/pi) * (x + 0.044715 * x^3))
    sqrt2_over_pi = 0.7978845608028654
    c = 0.044715

    a3 = a * a * a
    inner = sqrt2_over_pi * (a + c * a3)
    gelu_a = a * tl.sigmoid(2.0 * inner)

    # Gate
    out = gelu_a * b

    # Store result
    tl.store(out_row + col_offs, out, eviction_policy='evict_first')


# ---------------------------------------------------------------------------
# Public entry point – allocates output and launches the kernel.
# ---------------------------------------------------------------------------
def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    x: float32 tensor of shape [8192, 8192]
    returns: float32 tensor of shape [8192, 4096]
    """
    M, K = x.shape
    N = K // 2  # 4096 for the given input

    out = torch.empty(M, N, device=x.device, dtype=x.dtype)

    BLOCK_SIZE = N  # each block processes an entire output row
    grid = (M,)

    geglu_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=4,
    )

    return out