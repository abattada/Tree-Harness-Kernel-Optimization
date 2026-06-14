import torch
import triton
import triton.language as tl


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    Compute per-row population mean and variance using a single-pass sum-of-squares
    approach.  Each program processes several rows (ROWS_PER_PROG) to reduce launch
    overhead and improve occupancy.
    """
    N_ROWS, N_COLS = x.shape
    out = torch.empty(2, N_ROWS, dtype=x.dtype, device=x.device)

    # Tunable constants – balances occupancy vs. register pressure
    BLOCK_SIZE = 1024  # powers of two dividing N_COLS (4096)
    ROWS_PER_PROG = 4  # 2D tiling: each program handles a block of rows
    NUM_WARPS = 4
    NUM_STAGES = 2

    assert N_COLS % BLOCK_SIZE == 0, "BLOCK_SIZE must divide N_COLS for mask-free loop"
    assert N_ROWS % ROWS_PER_PROG == 0, "N_ROWS must be divisible by ROWS_PER_PROG"

    grid = (N_ROWS // ROWS_PER_PROG,)
    welford_kernel[grid](
        x, out,
        x.stride(0),
        N_ROWS=N_ROWS, N_COLS=N_COLS,
        BLOCK_SIZE=BLOCK_SIZE, ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return out


@triton.jit
def welford_kernel(x_ptr, out_ptr, stride_row,
                   N_ROWS: tl.constexpr, N_COLS: tl.constexpr,
                   BLOCK_SIZE: tl.constexpr, ROWS_PER_PROG: tl.constexpr):
    """
    Each program handles ROWS_PER_PROG consecutive rows.  Within each row it
    processes column tiles of size BLOCK_SIZE, accumulating partial (sum,
    sum_of_squares) at the tile level.
    """
    tl.static_assert(N_COLS % BLOCK_SIZE == 0,
                     "N_COLS must be divisible by BLOCK_SIZE")
    tl.static_assert(N_ROWS % ROWS_PER_PROG == 0,
                     "N_ROWS must be divisible by ROWS_PER_PROG")

    pid = tl.program_id(0)                  # block index
    row_start = pid * ROWS_PER_PROG

    # Per-row accumulators: arrays of length ROWS_PER_PROG
    acc_sum = tl.zeros([ROWS_PER_PROG], dtype=tl.float32)
    acc_sum_sq = tl.zeros([ROWS_PER_PROG], dtype=tl.float32)

    # Loop over rows assigned to this program
    for r in tl.static_range(0, ROWS_PER_PROG):
        row_base = x_ptr + (row_start + r) * stride_row
        row_base = tl.multiple_of(row_base, BLOCK_SIZE)

        # Process all column tiles for this row — no boundary mask needed
        for col_start in tl.static_range(0, N_COLS, BLOCK_SIZE):
            offsets = col_start + tl.arange(0, BLOCK_SIZE)
            vals = tl.load(row_base + offsets, cache_modifier='.cg')
            acc_sum[r] += tl.sum(vals, axis=0)
            acc_sum_sq[r] += tl.sum(vals * vals, axis=0)

    # Convert row length to float32 for division
    N = tl.cast(N_COLS, tl.float32)

    # Compute and store per-row mean and population variance
    for r in tl.static_range(0, ROWS_PER_PROG):
        row_idx = row_start + r
        mean = acc_sum[r] / N
        var = acc_sum_sq[r] / N - mean * mean
        tl.store(out_ptr + row_idx, mean)
        tl.store(out_ptr + N_ROWS + row_idx, var)