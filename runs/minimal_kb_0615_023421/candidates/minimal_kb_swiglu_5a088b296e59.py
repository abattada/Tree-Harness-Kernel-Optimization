import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel: SWiGLU
# Input:  x [M, K] with K even; K = 8192, M = 8192
# Output: out [M, K//2] = [M, 4096]
#
# We process one row per program. Since BLOCK_SIZE = N = 4096 exactly matches
# the chunk dimension, masks are unnecessary. All loads/stores are contiguous
# and aligned, achieving high bandwidth.
# ---------------------------------------------------------------------------

@triton.jit
def swiglu_kernel(
    x_ptr,                # pointer to input (2D flattened)
    out_ptr,              # pointer to output (2D flattened)
    stride_x_row,         # row stride of x (in elements) = K
    stride_out_row,       # row stride of out (in elements) = N
    BLOCK_SIZE: tl.constexpr,   # ≡ N = 4096
):
    # ---- 1. row index ----------------------------------------------------
    row = tl.program_id(0)

    # ---- 2. pointers for this row ----------------------------------------
    x_row = x_ptr + row * stride_x_row
    out_row = out_ptr + row * stride_out_row

    # ---- 3. column indices (no mask needed: BLOCK_SIZE == N) ------------
    col_offs = tl.arange(0, BLOCK_SIZE)

    # ---- 4. load a and b (both contiguous in one row, each read once) ----
    a = tl.load(x_row + col_offs, eviction_policy='evict_first')
    b = tl.load(x_row + col_offs + BLOCK_SIZE, eviction_policy='evict_first')

    # ---- 5. compute SiLU(a) = a * sigmoid(a) -----------------------------
    sig_a = tl.sigmoid(a)
    silu_a = a * sig_a

    # ---- 6. gating: silu(a) * b ------------------------------------------
    out = silu_a * b

    # ---- 7. store result, evict_first (won't be read again) ---------------
    tl.store(out_row + col_offs, out, eviction_policy='evict_first')


# ---------------------------------------------------------------------------
# Public entry point – allocates output and launches the kernel.
# ---------------------------------------------------------------------------

def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    x: float32 tensor of shape [M, K]  (K even)
    returns: float32 tensor of shape [M, K//2] (SWiGLU activation)
    """
    M, K = x.shape
    N = K // 2                          # output feature dimension

    # Allocate output
    out = torch.empty(M, N, device=x.device, dtype=torch.float32)

    # Launch configuration
    BLOCK_SIZE = N                      # 4096, exactly covers one row's a/b
    grid = (M,)                         # one program per row

    swiglu_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,                    # good balance occupancy vs. registers
        num_stages=4,                   # moderate pipelining
    )
    return out