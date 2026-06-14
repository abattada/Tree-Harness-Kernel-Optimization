import torch
import triton
import triton.language as tl

# Fixed shape; compile‑time constants
N_ROWS = 8192
N_COLS = 4096
BLOCK_SIZE = N_COLS               # one row fits in one block, no masking
ROWS_PER_PROG = 4                 # process multiple rows per program (multirow_per_program)
INV_N = 1.0 / N_COLS              # compile‑time reciprocal
NUM_WARPS = 4
NUM_STAGES = 2

@triton.jit
def welford_kernel(
    x_ptr,                        # [N_ROWS, N_COLS] input
    out_ptr,                      # [2, N_ROWS] output (row0=mean, row1=var)
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INV_N: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    group_id = tl.program_id(0)
    # Loop over rows inside this group
    for i in range(ROWS_PER_PROG):
        row_idx = group_id * ROWS_PER_PROG + i
        row_start = row_idx * n_cols

        # Contiguous, aligned offsets for the whole row
        offsets = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
        x_ptrs = x_ptr + row_start + offsets

        # Load the row; evict_first because data used only once
        x = tl.load(x_ptrs, eviction_policy='evict_first')

        # Single‑pass sum and sum of squares in fp32
        s  = tl.sum(x, axis=0).to(tl.float32)
        sq = tl.sum(x * x, axis=0).to(tl.float32)

        mean = s * INV_N
        var  = (sq * INV_N) - mean * mean   # population variance

        # Store to output [2, N_ROWS]
        tl.store(out_ptr + 0 * n_rows + row_idx, mean)
        tl.store(out_ptr + 1 * n_rows + row_idx, var)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    Compute per‑row mean and population variance of a 2‑D float32 tensor.

    Equivalent to:
        mean = x.mean(dim=-1)
        var  = x.var(dim=-1, unbiased=False)
        return torch.stack([mean, var])

    Input:  x shape (8192, 4096), float32, CUDA device.
    Output: out shape (2, 8192), float32.
    """
    assert x.is_cuda and x.dtype == torch.float32
    assert x.shape == (N_ROWS, N_COLS), f"Expected ({N_ROWS}, {N_COLS}) but got {x.shape}"

    out = torch.empty(2, N_ROWS, dtype=torch.float32, device=x.device)

    grid = (N_ROWS // ROWS_PER_PROG,)
    welford_kernel[grid](
        x, out,
        n_rows=N_ROWS,
        n_cols=N_COLS,
        BLOCK_SIZE=BLOCK_SIZE,
        INV_N=INV_N,
        ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return out