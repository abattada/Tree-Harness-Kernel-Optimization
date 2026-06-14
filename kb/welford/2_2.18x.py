import torch
import triton
import triton.language as tl

# Tunable parameters
BLOCK_SIZE = 1024   # Must divide 4096 (n_cols)
NUM_WARPS = 4
NUM_STAGES = 4

@triton.jit
def welford_kernel(
    x_ptr,        # [n_rows, n_cols] input
    out_ptr,      # [2, n_rows] output (row0=mean, row1=var)
    n_cols: int,  # number of columns (static: 4096)
    n_rows: int,  # unused but informative
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)  # row index

    # Starting offsets
    row_start = pid * n_cols

    # Accumulators (global statistics)
    global_n = 0.0
    global_mean = 0.0
    global_m2 = 0.0

    # Iterate over row tiles
    for col_start in range(0, n_cols, BLOCK_SIZE):
        offsets = col_start + tl.arange(0, BLOCK_SIZE)
        # We assume n_cols is a multiple of BLOCK_SIZE, so no mask needed.
        x = tl.load(x_ptr + row_start + offsets)

        # Block statistics
        block_sum = tl.sum(x, axis=0)
        block_sum_sq = tl.sum(x * x, axis=0)
        block_n = BLOCK_SIZE
        block_mean = block_sum / block_n
        block_m2 = block_sum_sq - block_mean * block_sum  # M2 = sum((x-mean)^2)

        # Combine block statistics into global using Welford's parallel formula
        delta = block_mean - global_mean
        total_n = global_n + block_n
        # For first block (global_n=0), avoid division by zero; formula still works.
        # global_mean updates linearly, global_m2 adds contribution.
        # Careful with NaN when global_n=0: delta^2 * (0 * block_n / total_n) = 0.
        global_mean = (global_n * global_mean + block_n * block_mean) / total_n
        global_m2 = global_m2 + block_m2 + delta * delta * global_n * block_n / total_n
        global_n = total_n

    # Population variance (unbiased=False -> divide by n)
    var = global_m2 / global_n

    # Write results
    tl.store(out_ptr + 0 * n_rows + pid, global_mean)
    tl.store(out_ptr + 1 * n_rows + pid, var)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32
    n_rows, n_cols = x.shape
    # Allocate output: (2, n_rows)
    out = torch.empty(2, n_rows, dtype=torch.float32, device=x.device)

    grid = (n_rows,)
    welford_kernel[grid](
        x, out, n_cols, n_rows,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return out