import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel: SwiGLU (silu gating)
# Input:  x [M, K]  with K even, e.g. K = 8192
# Output: out [M, K//2] = silu(a) * b  where a,b = x.chunk(2, -1)
#
# This version uses a persistent kernel approach: a fixed number of program
# instances (GRID_SIZE) each loop over multiple rows (grid‑stride loop).
# This reduces launch overhead and improves occupancy on the large row count.
# Blocks cover a full row (BLOCK_SIZE = N) so no column masks are needed.
# ---------------------------------------------------------------------------

@triton.jit
def swiglu_kernel_persistent(
    x_ptr,                # pointer to input (2D, row-major)
    out_ptr,              # pointer to output (2D, row-major)
    M,                    # total number of rows
    stride_x_row,         # row stride of x (in elements) = K
    stride_out_row,       # row stride of out (in elements) = N
    BLOCK_SIZE: tl.constexpr,   # = N = K//2
    GRID_SIZE: tl.constexpr,    # number of program instances (stride)
):
    # Persistent loop: start at pid, stride by GRID_SIZE
    row = tl.program_id(0)
    while row < M:
        # Pointers to this row
        x_row = x_ptr + row * stride_x_row
        out_row = out_ptr + row * stride_out_row

        # Column offsets (BLOCK_SIZE covers exactly one half)
        col_offs = tl.arange(0, BLOCK_SIZE)

        # Load a and b (both halves of the row)
        # Use evict_first because each element is read only once
        a = tl.load(x_row + col_offs, eviction_policy='evict_first')
        b = tl.load(x_row + col_offs + BLOCK_SIZE, eviction_policy='evict_first')

        # silu(x) = x * sigmoid(x)
        sig_a = tl.sigmoid(a)
        silu_a = a * sig_a

        # Gating
        out = silu_a * b

        # Store result (evict_first, won't be reused)
        tl.store(out_row + col_offs, out, eviction_policy='evict_first')

        # Advance to next row
        row += GRID_SIZE


# ---------------------------------------------------------------------------
# Public entry point – allocates output and launches the kernel.
# ---------------------------------------------------------------------------

def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    x: float32 tensor of shape [M, K]  (K even)
    returns: float32 tensor of shape [M, K//2]
    """
    M, K = x.shape
    N = K // 2                           # output feature dimension

    # Allocate output
    out = torch.empty(M, N, device=x.device, dtype=torch.float32)

    # Choose GRID_SIZE so that each program processes ~8 rows
    # (e.g., 1024 for M=8192).  Power of two for optimal scheduling.
    GRID_SIZE = 1024

    # Launch configuration
    BLOCK_SIZE = N                       # 4096 for the given shape
    grid = (GRID_SIZE,)                  # persistent kernel, fewer programs

    swiglu_kernel_persistent[grid](
        x, out,
        M,
        x.stride(0), out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        GRID_SIZE=GRID_SIZE,
        num_warps=8,                     # balanced occupancy/register usage
        num_stages=5,                    # slightly deeper pipeline
    )
    return out