import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# GeGLU kernel using fast sigmoid-based tanh for the GELU approximation.
# Input:  x [M, K] with K even (here [8192, 8192])
# Output: out [M, K//2] = [8192, 4096]
# One program per row; BLOCK_SIZE equals N (4096) so no masking is needed.
# ---------------------------------------------------------------------------

@triton.jit
def geglu_kernel(
    x_ptr,               # pointer to input (2D, flat)
    out_ptr,             # pointer to output (2D, flat)
    stride_x_row,        # row stride of x in elements (= K)
    stride_out_row,      # row stride of out in elements (= N)
    BLOCK_SIZE: tl.constexpr,   # must equal N (K//2)
):
    # Row index
    row = tl.program_id(0)

    # Column offsets for the whole output row (no mask required)
    col_offs = tl.arange(0, BLOCK_SIZE)

    # Row pointers
    x_row = x_ptr + row * stride_x_row
    out_row = out_ptr + row * stride_out_row

    # Load a (first half) and b (second half) – each read exactly once
    a = tl.load(x_row + col_offs, eviction_policy='evict_first')
    b = tl.load(x_row + col_offs + BLOCK_SIZE, eviction_policy='evict_first')

    # GELU formula: 0.5 * a * (1 + tanh(sqrt(2/pi) * (a + 0.044715 * a**3)))
    # Use the identity tanh(x) = 2*sigmoid(2*x) - 1 for a faster code path.
    sqrt2_over_pi = 0.7978845608028654
    c = 0.044715

    a3 = a * a * a
    inner = sqrt2_over_pi * (a + c * a3)
    tanh_val = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
    gelu_a = 0.5 * a * (1.0 + tanh_val)

    # Gate
    out = gelu_a * b

    # Store result – also evict_first since it will not be read again
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

    out = torch.empty(M, N, device=x.device, dtype=torch.float32)

    grid = (M,)                         # one program per row
    geglu_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        BLOCK_SIZE=N,
        num_warps=8,
        num_stages=4,
    )

    return out