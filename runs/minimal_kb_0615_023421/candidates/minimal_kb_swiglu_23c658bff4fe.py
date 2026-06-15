import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel: SWiGLU
# Input:  x [M, K] with K even; K = 8192, M = 8192
# Output: out [M, K//2] = [M, 4096]
#
# Improved by processing multiple rows per program (grid-stride loop) to
# reduce launch overhead. The block size equals the output dimension so no
# masking is required. Eviction hints and reduced num_stages keep occupancy
# high.
# ---------------------------------------------------------------------------

@triton.jit
def swiglu_kernel(
    x_ptr,                # pointer to input (2D flattened)
    out_ptr,              # pointer to output (2D flattened)
    stride_x_row,         # row stride of x (in elements) = K
    stride_out_row,       # row stride of out (in elements) = N
    num_rows,             # total number of rows M
    BLOCK_SIZE: tl.constexpr,   # ≡ N = 4096
    ROWS_PER_BLOCK: tl.constexpr,   # rows handled per program
):
    pid = tl.program_id(0)
    start_row = pid * ROWS_PER_BLOCK
    col_offs = tl.arange(0, BLOCK_SIZE)

    for r in range(ROWS_PER_BLOCK):
        row = start_row + r
        if row < num_rows:
            x_row = x_ptr + row * stride_x_row
            out_row = out_ptr + row * stride_out_row

            # Load a and b (both contiguous, no mask needed)
            a = tl.load(x_row + col_offs, eviction_policy='evict_first')
            b = tl.load(x_row + col_offs + BLOCK_SIZE, eviction_policy='evict_first')

            # Compute SiLU(a) = a * sigmoid(a)
            sig_a = tl.sigmoid(a)
            silu_a = a * sig_a

            # Gating: silu(a) * b
            out = silu_a * b

            # Store result
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

    # Launch configuration: process two rows per program to halve launches
    ROWS_PER_BLOCK = 2
    grid = (triton.cdiv(M, ROWS_PER_BLOCK),)

    swiglu_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        M,
        BLOCK_SIZE=N,
        ROWS_PER_BLOCK=ROWS_PER_BLOCK,
        num_warps=8,
        num_stages=3,   # slightly reduced from 4 to free registers for the loop
    )
    return out