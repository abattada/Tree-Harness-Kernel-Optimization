import torch
import triton
import triton.language as tl

# Tunable parameters for initial implementation
BLOCK_SIZE = 512      # 8 tiles per row (4096 / 512 = 8)
NUM_WARPS = 4
NUM_STAGES = 2

@triton.jit
def welford_kernel(
    x_ptr,          # [n_rows, n_cols] input
    out_ptr,        # [2, n_rows] output (row0=mean, row1=var)
    n_cols: int,    # number of columns (4096)
    n_rows: int,    # number of rows (8192) – used to compute output strides
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)  # row index

    row_start = pid * n_cols

    # Accumulators for Welford online algorithm
    n_total = 0.0
    mean = 0.0
    m2 = 0.0

    # Iterate over tiles of the row
    for col_start in range(0, n_cols, BLOCK_SIZE):
        offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_ptr + row_start + offsets, mask=mask)

        # Block statistics (sum, sum of squares, block size)
        block_sum = tl.sum(x, axis=0)
        block_n = tl.sum(mask.to(tl.float32), axis=0)
        block_mean = block_sum / block_n
        block_sq = tl.sum(x * x, axis=0)
        block_m2 = block_sq - block_mean * block_sum  # sum((x - block_mean)^2)

        # Merge block into global accumulators (Welford parallel formula)
        if n_total > 0:
            delta = block_mean - mean
            # Weighted update of mean
            mean = (mean * n_total + block_mean * block_n) / (n_total + block_n)
            # Combine M2
            m2 = m2 + block_m2 + delta * delta * n_total * block_n / (n_total + block_n)
            n_total = n_total + block_n
        else:
            mean = block_mean
            m2 = block_m2
            n_total = block_n

    # Population variance (unbiased=False)
    var = m2 / n_total

    # Store results: layout is (2, n_rows) with row-major contiguous storage
    tl.store(out_ptr + pid, mean)           # row 0, column pid
    tl.store(out_ptr + n_rows + pid, var)   # row 1, column pid


def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32
    n_rows, n_cols = x.shape
    # Verify that n_cols is a multiple of BLOCK_SIZE for clean masking
    assert n_cols % BLOCK_SIZE == 0, "n_cols must be multiple of BLOCK_SIZE"

    out = torch.empty(2, n_rows, dtype=torch.float32, device=x.device)

    grid = (n_rows,)
    welford_kernel[grid](
        x, out,
        n_cols, n_rows,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return out