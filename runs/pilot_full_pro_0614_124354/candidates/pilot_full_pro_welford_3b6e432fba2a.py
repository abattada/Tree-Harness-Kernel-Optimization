import torch
import triton
import triton.language as tl


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    Compute per-row population mean and variance using a single-pass sum-of-squares approach.

    Args:
        x: float32 tensor of shape (8192, 4096)

    Returns:
        A float32 tensor of shape (2, 8192) where out[0, i] = mean of row i
        and out[1, i] = population variance of row i.
    """
    N_ROWS, N_COLS = x.shape
    out = torch.empty(2, N_ROWS, dtype=x.dtype, device=x.device)

    # Tuning knobs – obvious places to adjust for performance
    BLOCK_SIZE = 256          # Must divide N_COLS exactly (4096 % 256 == 0)
    NUM_WARPS = 4
    NUM_STAGES = 2

    assert N_COLS % BLOCK_SIZE == 0, "BLOCK_SIZE must divide N_COLS for boundary-free loop"

    grid = (N_ROWS,)
    welford_kernel[grid](
        x, out,
        x.stride(0),
        N_ROWS=N_ROWS, N_COLS=N_COLS,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return out


@triton.jit
def welford_kernel(x_ptr, out_ptr, stride_row,
                   N_ROWS: tl.constexpr, N_COLS: tl.constexpr,
                   BLOCK_SIZE: tl.constexpr):
    """
    Each program handles one row entirely.  The per-row computation uses a
    vectorized partial reduction of (sum, sum_of_squares) over static column
    blocks, followed by a final scalar mean/variance calculation.
    """
    tl.static_assert(N_COLS % BLOCK_SIZE == 0,
                     "N_COLS must be divisible by BLOCK_SIZE for this kernel")

    pid = tl.program_id(0)                 # row index
    row_base = x_ptr + pid * stride_row
    row_base = tl.multiple_of(row_base, BLOCK_SIZE)

    # Accumulators for online sum / sum of squares (fp32)
    acc_sum = tl.zeros([], dtype=tl.float32)
    acc_sum_sq = tl.zeros([], dtype=tl.float32)

    # Fully unrolled loop over column tiles – no boundary mask needed
    for col_start in tl.static_range(0, N_COLS, BLOCK_SIZE):
        offsets = col_start + tl.arange(0, BLOCK_SIZE)
        vals = tl.load(row_base + offsets, cache_modifier='.cg')
        # Reductions over the tile: sum and sum of squares
        acc_sum += tl.sum(vals, axis=0)
        acc_sum_sq += tl.sum(vals * vals, axis=0)

    # Convert row length to float for division
    N = tl.cast(N_COLS, tl.float32)
    mean = acc_sum / N
    var = acc_sum_sq / N - mean * mean          # population variance

    # Store results into (2, N_ROWS) contiguous output
    tl.store(out_ptr + pid, mean)
    tl.store(out_ptr + N_ROWS + pid, var)