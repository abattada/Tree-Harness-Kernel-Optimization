import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ---------------------------------------------------------------------------
# Triton kernel: GeGLU (gated GELU with tanh approximation), specialized for
# input shape (8192, 8192) -> output (8192, 4096).
#
# Key optimizations:
# - constexpr strides to fold pointer arithmetic (STRIDE_X=8192, STRIDE_OUT=4096)
# - full-row tile (BLOCK_SIZE=4096) eliminates masks
# - eviction hints for streaming data
# - tuned num_warps/num_stages for Blackwell occupancy
# ---------------------------------------------------------------------------

@triton.jit
def geglu_kernel(
    x_ptr,                              # pointer to input (2D)
    out_ptr,                            # pointer to output (2D)
    STRIDE_X: tl.constexpr,             # = 8192 (input rows stride in elements)
    STRIDE_OUT: tl.constexpr,           # = 4096 (output rows stride in elements)
    BLOCK_SIZE: tl.constexpr,           # = 4096 (output dimension)
):
    # Static assertion to guarantee no partial tiles
    tl.static_assert(BLOCK_SIZE == 4096, "BLOCK_SIZE must be 4096 for this specialization")

    # Row index
    row = tl.program_id(0)

    # Column offsets (0..BLOCK_SIZE-1)
    col_offs = tl.arange(0, BLOCK_SIZE)

    # Pointers to this row's data
    x_row = x_ptr + row * STRIDE_X
    out_row = out_ptr + row * STRIDE_OUT

    # Load a (first half) and b (second half) – evict_first because each is read once
    a = tl.load(x_row + col_offs, eviction_policy='evict_first')
    b = tl.load(x_row + col_offs + BLOCK_SIZE, eviction_policy='evict_first')

    # Compute GELU with tanh approximation
    # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    coeff1 = 0.7978845608028654   # sqrt(2/pi)
    coeff2 = 0.044715

    a3 = a * a * a
    inner = coeff1 * (a + coeff2 * a3)
    tanh_val = libdevice.tanh(inner)
    gelu_a = 0.5 * a * (1.0 + tanh_val)

    # Gating: out = gelu(a) * b
    out = gelu_a * b

    # Store result – evict_first since it is written once and not read again
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
    N = K // 2   # 4096

    # Allocate output
    out = torch.empty(M, N, device=x.device, dtype=torch.float32)

    # Launch configuration (hardcoded for known shapes)
    grid = (M,)
    geglu_kernel[grid](
        x, out,
        STRIDE_X=8192,
        STRIDE_OUT=4096,
        BLOCK_SIZE=4096,
        num_warps=8,
        num_stages=5,           # slightly more pipeline stages for better overlap
    )

    return out