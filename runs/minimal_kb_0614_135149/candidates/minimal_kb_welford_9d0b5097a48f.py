import torch
import triton
import triton.language as tl

# Fixed shape for the operator
N_ROWS = 8192
N_COLS = 4096

# Entire row fits in one block, no masking needed
BLOCK_SIZE = N_COLS

# Process multiple rows per program to reduce scheduling overhead
ROWS_PER_PROG = 8

# Compile‑time reciprocal for faster division
INV_N = 1.0 / N_COLS


@triton.jit
def welford_kernel(
    x_ptr,
    out_ptr,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
    INV_N: tl.constexpr,
):
    pid = tl.program_id(0)
    base_row = pid * ROWS_PER_PROG

    # Sequential processing of multiple rows amortises launch overhead
    # and keeps the memory pipeline busy without extra kernel launches.
    for r in tl.range(0, ROWS_PER_PROG):
        row = base_row + r
        row_start = row * n_cols

        # Vectorised load of the whole row
        offsets = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
        x = tl.load(x_ptr + row_start + offsets, eviction_policy='evict_first')

        # Single‑pass Welford sums
        s = tl.sum(x, axis=0).to(tl.float32)
        sq = tl.sum(x * x, axis=0).to(tl.float32)

        mean = s * INV_N
        var = (sq * INV_N) - mean * mean   # population variance

        # Store results into [2, n_rows] layout
        tl.store(out_ptr + 0 * n_rows + row, mean)
        tl.store(out_ptr + 1 * n_rows + row, var)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    Compute per‑row mean and population variance of a float32 tensor.

    Equivalent to:
        mean = x.mean(dim=-1)
        var  = x.var(dim=-1, unbiased=False)
        return torch.stack([mean, var])
    """
    assert x.is_cuda and x.dtype == torch.float32
    assert x.shape == (N_ROWS, N_COLS), f"Expected ({N_ROWS}, {N_COLS}) but got {x.shape}"

    out = torch.empty((2, N_ROWS), dtype=torch.float32, device=x.device)

    grid = (N_ROWS // ROWS_PER_PROG,)
    welford_kernel[grid](
        x, out,
        n_rows=N_ROWS,
        n_cols=N_COLS,
        BLOCK_SIZE=BLOCK_SIZE,
        ROWS_PER_PROG=ROWS_PER_PROG,
        INV_N=INV_N,
        num_warps=4,
        num_stages=2,         # no pipelining needed; low stages reduce register pressure
    )
    return out