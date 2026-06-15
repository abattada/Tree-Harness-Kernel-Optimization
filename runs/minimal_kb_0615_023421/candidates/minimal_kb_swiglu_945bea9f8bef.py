import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel: SwiGLU (silu gating)
# Input:  x [M, K]  with K even, e.g. K = 8192
# Output: out [M, K//2] = silu(a) * b  where a,b = x.chunk(2, -1)
#
# Optimizations:
#   - Grid-stride loop: each program processes multiple rows to reduce
#     launch overhead and improve L2 reuse.
#   - Constexpr specialization: K and N are compile-time constants;
#     stride hints (max_contiguous, multiple_of) enable wider vector loads.
#   - Eviction hints: streaming loads/stores use evict_first.
# ---------------------------------------------------------------------------

@triton.jit
def swiglu_kernel(
    x_ptr,                # pointer to input (2D, row-major)
    out_ptr,              # pointer to output (2D, row-major)
    M,                    # number of rows
    stride_x_row,         # row stride of x (in elements) = K
    stride_out_row,       # row stride of out (in elements) = N
    BLOCK_SIZE: tl.constexpr,   # = N = K//2
    ROWS_PER_PROG: tl.constexpr, # number of rows processed per program
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROG

    # Offsets within a row (constant for all rows in this block)
    col_offs = tl.arange(0, BLOCK_SIZE)

    # Loop over the assigned rows
    for r in range(ROWS_PER_PROG):
        row = row_start + r
        if row < M:
            # Pointers for this row
            x_row = x_ptr + row * stride_x_row
            out_row = out_ptr + row * stride_out_row

            # Load a and b with alignment hints for vectorization
            a = tl.load(x_row + col_offs, eviction_policy='evict_first')
            b = tl.load(x_row + col_offs + BLOCK_SIZE, eviction_policy='evict_first')

            # silu(x) = x * sigmoid(x)
            sig_a = tl.sigmoid(a)
            silu_a = a * sig_a

            # Gating
            out = silu_a * b

            # Store result (streaming write)
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
    N = K // 2                           # output feature dimension

    # Allocate output
    out = torch.empty(M, N, device=x.device, dtype=torch.float32)

    # Launch configuration: each program processes 4 rows (grid-stride loop)
    ROWS_PER_PROG = 4
    grid = (triton.cdiv(M, ROWS_PER_PROG),)

    # Shape constants (passed as scalars to kernel; stride is deduced)
    swiglu_kernel[grid](
        x, out,
        M,
        x.stride(0), out.stride(0),
        BLOCK_SIZE=N,
        ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=8,                    # balanced occupancy/register usage
        num_stages=4,                   # moderate pipeline depth
    )
    return out