import torch
import triton
import triton.language as tl

# Tunable parameters for the reference kernel
BLOCK_SIZE = 1024        # Must divide 4096 (power of two)
NUM_WARPS = 4
NUM_STAGES = 4

@triton.jit
def welford_kernel(
    x_ptr,        # [n_rows, n_cols] input
    out_ptr,      # [2, n_rows] output (row0=mean, row1=var)
    n_rows: int,
    n_cols: int,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)  # row index

    # Accumulators for the whole row
    global_n = 0.0
    global_mean = 0.0
    global_m2 = 0.0

    # Iterate over row in blocks
    for col_start in range(0, n_cols, BLOCK_SIZE):
        offsets = col_start + tl.arange(0, BLOCK_SIZE)
        # Mask for partial last block (though n_cols is always multiple of BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_ptr + pid * n_cols + offsets, mask=mask, other=0.0)

        # Compute block statistics
        block_sum = tl.sum(x, axis=0)
        block_sum_sq = tl.sum(x * x, axis=0)
        block_n = tl.sum(mask.to(tl.float32))  # actual number of elements in block
        block_mean = block_sum / block_n
        block_m2 = block_sum_sq - block_mean * block_sum

        # Combine block into global using Welford parallel formula
        # Guard against division by zero when global_n == 0 (first block)
        delta = block_mean - global_mean
        total_n = global_n + block_n
        # Update global mean
        global_mean = tl.where(global_n > 0,
                               (global_n * global_mean + block_n * block_mean) / total_n,
                               block_mean)
        # Update global sum of squared differences
        # When global_n == 0, the correction term is zero
        correction = delta * delta * global_n * block_n / total_n
        global_m2 = global_m2 + block_m2 + correction
        global_n = total_n

    # Population variance = M2 / n
    var = global_m2 / global_n

    # Store results
    tl.store(out_ptr + 0 * n_rows + pid, global_mean)
    tl.store(out_ptr + 1 * n_rows + pid, var)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32
    n_rows, n_cols = x.shape
    out = torch.empty(2, n_rows, dtype=torch.float32, device=x.device)

    grid = (n_rows,)
    welford_kernel[grid](
        x, out, n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return out