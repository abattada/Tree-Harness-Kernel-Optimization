import torch
import triton
import triton.language as tl

# Tunable parameters
BLOCK_SIZE = 4096   # Full row (n_cols)
NUM_WARPS = 4
NUM_STAGES = 4
ROWS_PER_PROG = 4   # Process multiple rows per program to amortize overhead

@triton.jit
def welford_kernel(
    x_ptr,                          # input: (n_rows, n_cols)
    out_ptr,                        # output: (2, n_rows)
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    # Block index -> base row
    pid = tl.program_id(0)
    base_row = pid * ROWS_PER_PROG

    # Loop over rows in this tile
    for row_offset in range(ROWS_PER_PROG):
        row_idx = base_row + row_offset
        # Check if this row is valid (mask)
        if row_idx < n_rows:
            row_start = row_idx * n_cols
            # Load full row (no mask needed because n_cols divides BLOCK_SIZE)
            offsets = tl.arange(0, BLOCK_SIZE)
            x = tl.load(x_ptr + row_start + offsets)

            # Sum and sum of squares in float32 for precision
            s = tl.sum(x, axis=0).to(tl.float32)
            sq = tl.sum(x * x, axis=0).to(tl.float32)

            mean = s / n_cols
            var = (sq / n_cols) - mean * mean   # population variance

            # Store results (output layout: [2, n_rows])
            out_ptr_mean = out_ptr + 0 * n_rows + row_idx
            out_ptr_var  = out_ptr + 1 * n_rows + row_idx
            tl.store(out_ptr_mean, mean)
            tl.store(out_ptr_var, var)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32
    n_rows, n_cols = x.shape

    # Output: row 0 = mean, row 1 = population variance
    out = torch.empty(2, n_rows, dtype=torch.float32, device=x.device)

    # Grid covers all rows, grouped into batches of ROWS_PER_PROG
    grid = ((n_rows + ROWS_PER_PROG - 1) // ROWS_PER_PROG,)

    welford_kernel[grid](
        x,
        out,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return out