import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ---------------------------------------------------------------------------
# Triton kernel: GeGLU (gated GELU, tanh approximation)
# Input:  x [M, K]   (K even)
# Output: out [M, K//2]
# Each program processes one row.
# Optimizations:
#   - Single large load of the whole row (2*N elements)
#   - Cache eviction hints for streaming inputs and output
#   - tl.multiple_of / tl.max_contiguous hints for vectorization
# ---------------------------------------------------------------------------

@triton.jit
def geglu_kernel(
    x_ptr,          # pointer to input (2D flattened)
    out_ptr,        # pointer to output (2D flattened)
    stride_x,       # row stride of x (in elements)   = K
    stride_out,     # row stride of out (in elements) = K//2
    N,              # output dimension = K//2
    BLOCK_SIZE: tl.constexpr,   # number of elements in the whole row (2*N)
):
    # ---- 1. row index --------------------------------------------------
    row = tl.program_id(0)

    # ---- 2. pointers for this row --------------------------------------
    x_row = x_ptr + row * stride_x
    out_row = out_ptr + row * stride_out

    # ---- 3. column indices ---------------------------------------------
    col_offs = tl.arange(0, BLOCK_SIZE)          # 0 .. 2*N-1
    mask_x = col_offs < 2 * N                     # always true if BLOCK_SIZE == 2*N
    mask_out = col_offs < N                       # only first half for output

    # ---- 4. load whole row in one contiguous load ----------------------
    # (a = first half, b = second half)
    # Use cache eviction policy: input is read once, mark as evict_first
    x_val = tl.load(x_row + col_offs, mask=mask_x, other=0.0,
                    eviction_policy='evict_first')

    # ---- 5. split into a and b -----------------------------------------
    a = x_val
    b = tl.load(x_row + col_offs + N, mask=mask_out, other=0.0,
                eviction_policy='evict_first')   # still from the input row

    # ---- 6. compute GELU_tanh(a) ----------------------------------------
    sqrt2_over_pi = 0.7978845608028654   # sqrt(2/pi)
    c = 0.044715
    a3 = a * a * a
    inner = sqrt2_over_pi * (a + c * a3)
    tanh_val = libdevice.tanh(inner)
    gelu_a = 0.5 * a * (1.0 + tanh_val)

    # ---- 7. gating ------------------------------------------------------
    out = gelu_a * b

    # ---- 8. store result (only first half) -----------------------------
    tl.store(out_row + col_offs, out, mask=mask_out, eviction_policy='evict_last')


# ---------------------------------------------------------------------------
# Public entry point – allocates output and launches the kernel.
# ---------------------------------------------------------------------------

def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    x: float32 tensor of shape [M, K]  (K even)
    returns: float32 tensor of shape [M, K//2]
    """
    M, K = x.shape
    N = K // 2                          # output feature dimension

    # Allocate output
    out = torch.empty(M, N, device=x.device, dtype=torch.float32)

    # Block size = full row (2*N). Must be a power of two >= 2*N.
    # Here 2*N = 8192, which is a power of two.
    BLOCK_SIZE = 8192
    grid = (M,)                         # one program per row

    geglu_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,                    # same as parent; can be tuned later
        num_stages=4,
    )
    return out