import torch
import triton
import triton.language as tl


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    Compute per-row population mean and variance using a multi-row kernel:
    each program processes ROWS_PER_PROG consecutive rows, amortising launch
    overhead and better utilising the SM for the long row dimension.

    Args:
        x: float32 tensor of shape (8192, 4096)

    Returns:
        A float32 tensor of shape (2, 8192) where out[0, i] = mean of row i
        and out[1, i] = population variance of row i.
    """
    N_ROWS, N_COLS = x.shape
    out = torch.empty(2, N_ROWS, dtype=x.dtype, device=x.device)

    # Tuning knobs – adjusted for multi-row kernel
    BLOCK_SIZE = 512          # must divide N_COLS exactly (4096 % 512 == 0)
    ROWS_PER_PROG = 4         # each program handles 4 consecutive rows
    NUM_WARPS = 8
    NUM_STAGES = 4

    assert N_COLS % BLOCK_SIZE == 0, "BLOCK_SIZE must divide N_COLS"
    assert N_ROWS % ROWS_PER_PROG == 0, "ROWS_PER_PROG must divide N_ROWS"

    # One program per ROWS_PER_PROG rows
    grid = (N_ROWS // ROWS_PER_PROG,)
    welford_multirow_kernel[grid](
        x, out,
        x.stride(0),
        N_ROWS=N_ROWS, N_COLS=N_COLS,
        ROWS_PER_PROG=ROWS_PER_PROG,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return out


@triton.jit
def welford_multirow_kernel(x_ptr, out_ptr, stride_row,
                            N_ROWS: tl.constexpr, N_COLS: tl.constexpr,
                            ROWS_PER_PROG: tl.constexpr,
                            BLOCK_SIZE: tl.constexpr):
    """
    Multi‑row reduction kernel. Each program handles ROWS_PER_PROG
    consecutive rows, accumulating sum and sum of squares across column
    tiles. The final mean and variance are written to the output.
    """
    tl.static_assert(N_COLS % BLOCK_SIZE == 0,
                     "N_COLS must be divisible by BLOCK_SIZE")

    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROG

    # Per‑row accumulators (fp32)
    acc_sum   = tl.zeros((ROWS_PER_PROG,), dtype=tl.float32)
    acc_sum_sq = tl.zeros((ROWS_PER_PROG,), dtype=tl.float32)

    # Iterate over column tiles (static loop, fully unrolled)
    for col_start in tl.static_range(0, N_COLS, BLOCK_SIZE):
        offsets = col_start + tl.arange(0, BLOCK_SIZE)

        # For each row assigned to this program, accumulate tile sums
        for j in range(ROWS_PER_PROG):
            row = row_start + j
            if row < N_ROWS:   # always true when grid is exact divisor
                row_base = x_ptr + row * stride_row
                vals = tl.load(row_base + offsets, cache_modifier='.cg')
                s  = tl.sum(vals)
                s2 = tl.sum(vals * vals)

                # Update the j‑th accumulator (using a one‑hot mask)
                mask_j = tl.arange(0, ROWS_PER_PROG) == j
                acc_sum   += mask_j * s
                acc_sum_sq += mask_j * s2

    # Population variance:  E[X^2] - (E[X])^2
    N = tl.cast(N_COLS, tl.float32)
    mean = acc_sum / N
    var  = acc_sum_sq / N - mean * mean

    # Write results for each handled row
    for j in range(ROWS_PER_PROG):
        row = row_start + j
        if row < N_ROWS:
            tl.store(out_ptr + row,          mean[j])
            tl.store(out_ptr + N_ROWS + row, var[j])