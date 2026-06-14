import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Optimized Triton kernel for per‑row mean and population variance
# (Welford‑style single‑pass sum/sumsq).
#
# Improvement over the parent (round 0):
#   • Multi‑row per program – each block processes ROWS_PER_PROG consecutive
#     rows sequentially, reducing the grid from 8192 to 256 and thereby
#     cutting launch overhead significantly. The row‑wise reduction itself
#     is unchanged (entire row loaded in one block, fully coalesced).
#   • Compile‑time constants for everything, eviction hints, and a lean
#     two‑stage pipeline tuning.
# ---------------------------------------------------------------------------

N_ROWS = 8192
N_COLS = 4096
BLOCK_SIZE = N_COLS
ROWS_PER_PROG = 32
INV_N = 1.0 / N_COLS

@triton.jit
def welford_kernel(
    x_ptr,                     # [n_rows, n_cols] float32
    out_ptr,                   # [2, n_rows] float32 (row0=mean, row1=var)
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
    INV_N: tl.constexpr,
):
    pid = tl.program_id(0)
    base_row = pid * ROWS_PER_PROG

    # Offsets for one full row – no masking needed (n_cols divides BLOCK_SIZE)
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)

    # Process up to ROWS_PER_PROG rows inside this block
    for r in range(ROWS_PER_PROG):
        row_id = base_row + r
        if row_id >= n_rows:
            break  # last block may have fewer rows

        row_start = row_id * n_cols
        x_ptrs = x_ptr + row_start + offsets

        # Stream the row once
        x = tl.load(x_ptrs, eviction_policy='evict_first')

        # Single‑pass sums in fp32
        s  = tl.sum(x, axis=0).to(tl.float32)
        sq = tl.sum(x * x, axis=0).to(tl.float32)

        # Population mean & variance (unbiased = False)
        mean = s * INV_N
        var  = (sq * INV_N) - mean * mean

        # Write results to output layout [2, n_rows]
        tl.store(out_ptr + 0 * n_rows + row_id, mean)
        tl.store(out_ptr + 1 * n_rows + row_id, var)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    Compute per‑row mean and population variance of a (8192, 4096) float32 tensor.

    Equivalent to:
        mean = x.mean(dim=-1)
        var  = x.var(dim=-1, unbiased=False)
        return torch.stack([mean, var])

    Returns:
        out: (2, 8192) float32 tensor. Row 0 = means, row 1 = variances.
    """
    assert x.is_cuda and x.dtype == torch.float32
    assert x.shape == (N_ROWS, N_COLS), \
        f"Expected shape ({N_ROWS}, {N_COLS}), got {x.shape}"

    out = torch.empty((2, N_ROWS), dtype=torch.float32, device=x.device)

    grid = (triton.cdiv(N_ROWS, ROWS_PER_PROG),)
    welford_kernel[grid](
        x,
        out,
        n_rows=N_ROWS,
        n_cols=N_COLS,
        BLOCK_SIZE=BLOCK_SIZE,
        ROWS_PER_PROG=ROWS_PER_PROG,
        INV_N=INV_N,
        num_warps=4,
        num_stages=2,
    )
    return out