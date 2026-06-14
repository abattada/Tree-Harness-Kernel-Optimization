import torch
import triton
import triton.language as tl


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    Compute per-row population mean and variance using a single‑pass sum‑of‑squares
    approach with a multirow‑per‑program grid to reduce launch overhead and improve
    SM utilisation.
    """
    N_ROWS, N_COLS = x.shape
    out = torch.empty(2, N_ROWS, dtype=x.dtype, device=x.device)

    # Tuned parameters – take advantage of multirow processing
    BLOCK_SIZE = 512          # power of two, divides 4096 exactly
    ROWS_PER_PROG = 8         # each program processes 8 rows
    NUM_WARPS = 8
    NUM_STAGES = 2

    assert N_COLS % BLOCK_SIZE == 0, "BLOCK_SIZE must divide N_COLS"
    assert N_ROWS % ROWS_PER_PROG == 0, "N_ROWS must be divisible by ROWS_PER_PROG"

    grid = (N_ROWS // ROWS_PER_PROG,)
    welford_kernel[grid](
        x, out,
        x.stride(0),
        N_ROWS=N_ROWS, N_COLS=N_COLS,
        BLOCK_SIZE=BLOCK_SIZE,
        ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return out


@triton.jit
def welford_kernel(x_ptr, out_ptr, stride_row,
                   N_ROWS: tl.constexpr, N_COLS: tl.constexpr,
                   BLOCK_SIZE: tl.constexpr, ROWS_PER_PROG: tl.constexpr):
    """
    Each program handles ROWS_PER_PROG contiguous rows, computing per‑row
    mean and population variance in a fully vectorised single pass.
    """
    tl.static_assert(N_COLS % BLOCK_SIZE == 0,
                     "N_COLS must be divisible by BLOCK_SIZE for mask‑free loop")

    pid = tl.program_id(0)          # block index over row groups

    # Iterate over the chunk of rows assigned to this program
    for r in tl.static_range(ROWS_PER_PROG):
        row = pid * ROWS_PER_PROG + r
        # Graceful exit for the last group (N_ROWS was verified divisible)
        if row >= N_ROWS:
            break

        row_base = x_ptr + row * stride_row
        row_base = tl.multiple_of(row_base, BLOCK_SIZE)

        acc_sum = tl.zeros([], dtype=tl.float32)
        acc_sum_sq = tl.zeros([], dtype=tl.float32)

        # Fully unrolled inner loop – every column element is covered exactly
        for col_start in tl.static_range(0, N_COLS, BLOCK_SIZE):
            offsets = col_start + tl.arange(0, BLOCK_SIZE)
            vals = tl.load(row_base + offsets, cache_modifier='.cg')
            acc_sum += tl.sum(vals, axis=0)
            acc_sum_sq += tl.sum(vals * vals, axis=0)

        N = tl.cast(N_COLS, tl.float32)
        mean = acc_sum / N
        var = acc_sum_sq / N - mean * mean   # population variance

        tl.store(out_ptr + row, mean)
        tl.store(out_ptr + N_ROWS + row, var)