import torch
import triton
import triton.language as tl

# Tunable parameters — good starting point for 4096 columns
BLOCK_SIZE = 4096   # equal to n_cols, no masking needed
NUM_WARPS = 4
NUM_STAGES = 4

@triton.jit
def welford_kernel(
    x_ptr,          # input: (n_rows, n_cols)
    out_ptr,        # output: (2, n_rows)
    n_cols: int,    # number of columns (static: 4096)
    n_rows: int,    # number of rows (for stride)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)          # row index
    row_start = pid * n_cols

    # Load the entire row – no mask needed because BLOCK_SIZE divides n_cols
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + row_start + offsets)

    # Single‑pass sums in float32 for precision
    s = tl.sum(x, axis=0).to(tl.float32)
    sq = tl.sum(x * x, axis=0).to(tl.float32)

    mean = s / n_cols
    # Population variance: E[X^2] - (E[X])^2
    var = (sq / n_cols) - mean * mean

    # Output layout: [2, n_rows]
    out_ptr_mean = out_ptr + 0 * n_rows + pid
    out_ptr_var  = out_ptr + 1 * n_rows + pid
    tl.store(out_ptr_mean, mean)
    tl.store(out_ptr_var, var)

def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32
    n_rows, n_cols = x.shape

    # Output: row 0 = mean, row 1 = population variance
    out = torch.empty(2, n_rows, dtype=torch.float32, device=x.device)

    grid = (n_rows,)
    welford_kernel[grid](
        x,
        out,
        n_cols=n_cols,
        n_rows=n_rows,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return out