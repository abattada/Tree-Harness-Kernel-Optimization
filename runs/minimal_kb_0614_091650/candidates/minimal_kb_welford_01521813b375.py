import torch
import triton
import triton.language as tl

# Use a persistent kernel: #programs = 80 (close to #SMs on RTX 5090),
# each program processes multiple rows in a grid-stride loop.
# This reduces launch overhead and improves occupancy.
GRID_SIZE = 80  # constant, will be used as tl.constexpr

@triton.jit
def welford_kernel(
    x_ptr,          # input: [n_rows, n_cols]
    out_ptr,        # output: [2, n_rows] (row0=mean, row1=population variance)
    n_rows: int,    # total number of rows
    n_cols: int,    # number of columns (static: 4096)
    BLOCK_SIZE: tl.constexpr,
    GRID_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    # Each program processes rows pid, pid+GRID_SIZE, pid+2*GRID_SIZE, ...
    for i in range(pid, n_rows, GRID_SIZE):
        row_start = i * n_cols

        # Load full row (no mask needed because n_cols == BLOCK_SIZE)
        offsets = tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + row_start + offsets)

        # Single-pass sum and sum of squares
        s = tl.sum(x, axis=0).to(tl.float32)
        sq = tl.sum(x * x, axis=0).to(tl.float32)

        mean = s / n_cols
        var = (sq / n_cols) - mean * mean   # population variance

        # Store results
        out_ptr_mean = out_ptr + 0 * n_rows + i
        out_ptr_var  = out_ptr + 1 * n_rows + i
        tl.store(out_ptr_mean, mean)
        tl.store(out_ptr_var, var)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32
    n_rows, n_cols = x.shape

    # Output: row 0 = mean, row 1 = population variance
    out = torch.empty(2, n_rows, dtype=torch.float32, device=x.device)

    BLOCK_SIZE = n_cols  # 4096, full row per load

    grid = (GRID_SIZE,)
    welford_kernel[grid](
        x,
        out,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        GRID_SIZE=GRID_SIZE,
    )
    return out