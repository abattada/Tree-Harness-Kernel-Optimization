import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel: SwiGLU (silu gating)
# Input:  x [M, K]  with K even, e.g. K = 8192
# Output: out [M, K//2] = silu(a) * b  where a,b = x.chunk(2, -1)
#
# Optimized for RTX 5090:
#  - Full-row blocks (BLOCK_SIZE = N) to avoid masking.
#  - Constexpr hints for alignment (multiple_of) so compiler can vectorize.
#  - 8 warps for good occupancy / register balance; 4 stages for pipelining.
#  - Eviction policy 'evict_first' for streaming loads/stores.
# ---------------------------------------------------------------------------

@triton.jit
def swiglu_kernel(
    x_ptr,                # pointer to input (2D, row-major)
    out_ptr,              # pointer to output (2D, row-major)
    stride_x_row,         # row stride of x (in elements) = K
    stride_out_row,       # row stride of out (in elements) = N
    BLOCK_SIZE: tl.constexpr,   # = N = K//2
):
    pid = tl.program_id(0)

    # Row pointer, aligned to 256-byte boundary (K*4 is multiple of 32768)
    x_row = tl.multiple_of(x_ptr + pid * stride_x_row, 256)
    out_row = tl.multiple_of(out_ptr + pid * stride_out_row, 256)

    col_offs = tl.arange(0, BLOCK_SIZE)

    # Load a and b (both halves of the row)
    a = tl.load(x_row + col_offs, eviction_policy='evict_first')
    b = tl.load(x_row + col_offs + BLOCK_SIZE, eviction_policy='evict_first')

    # silu(x) = x * sigmoid(x)
    silu_a = a * tl.sigmoid(a)

    # Gating
    out = silu_a * b

    # Store result
    tl.store(out_row + col_offs, out, eviction_policy='evict_first')


# ---------------------------------------------------------------------------
# Public entry point – allocates output and launches the kernel.
# ---------------------------------------------------------------------------

def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    x: float32 tensor of shape [M, K]  (K even)
    returns: float32 tensor of shape [M, K//2]
    """
    M, K = x.shape
    N = K // 2

    out = torch.empty(M, N, device=x.device, dtype=torch.float32)

    BLOCK_SIZE = N
    grid = (M,)

    swiglu_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,           # balanced occupancy / register use on Blackwell
        num_stages=4,          # moderate pipeline depth
    )
    return out