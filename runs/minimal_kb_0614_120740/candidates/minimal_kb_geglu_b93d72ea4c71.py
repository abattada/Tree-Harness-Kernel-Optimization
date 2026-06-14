import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ---------------------------------------------------------------------------
# Triton kernel: GeGLU (gated GELU with tanh approximation)
# Input:  x [M, K] with K even, K = 2 * N, N = 4096
# Output: out [M, N]
#
# One program per row, BLOCK_SIZE = N, no masks needed.
# Strides are compile-time constants derived from N, enabling simpler addressing.
# ---------------------------------------------------------------------------

@triton.jit
def geglu_kernel(
    x_ptr,                # pointer to input (2D flattened)
    out_ptr,              # pointer to output (2D flattened)
    N: tl.constexpr,      # output dimension = K//2
    BLOCK_SIZE: tl.constexpr,   # must equal N
):
    tl.static_assert(BLOCK_SIZE == N,
                     "BLOCK_SIZE must equal N for this specialization")

    # Row index
    row = tl.program_id(0)

    # Compute row pointers from N (compile-time constant)
    x_row = x_ptr + row * (2 * N)      # stride_x = 2 * N
    out_row = out_ptr + row * N        # stride_out = N

    # Column offsets (full block, no mask)
    col_offs = tl.arange(0, BLOCK_SIZE)

    # Load a (first half) and b (second half)
    a = tl.load(x_row + col_offs, eviction_policy='evict_first')
    b = tl.load(x_row + col_offs + N, eviction_policy='evict_first')

    # GELU tanh approximation
    sqrt2_over_pi = 0.7978845608028654
    c = 0.044715

    a3 = a * a * a
    inner = sqrt2_over_pi * (a + c * a3)
    tanh_val = libdevice.tanh(inner)
    gelu_a = 0.5 * a * (1.0 + tanh_val)

    # Gating
    out = gelu_a * b

    # Store output
    tl.store(out_row + col_offs, out, eviction_policy='evict_first')


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    x: float32 tensor of shape [M, K]  (K even)
    returns: float32 tensor of shape [M, K//2]
    """
    M, K = x.shape
    N = K // 2

    out = torch.empty(M, N, device=x.device, dtype=torch.float32)

    BLOCK_SIZE = N   # 4096
    grid = (M,)

    geglu_kernel[grid](
        x, out,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=4,
    )
    return out