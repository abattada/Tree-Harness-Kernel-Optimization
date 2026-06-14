import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel: per‑row mean and population variance (single‑pass sum/sumsq)
# with multi‑row processing to reduce launch overhead.
#
# Each program processes ROWS_PER_PROG rows sequentially, reducing the grid
# from 8192 to 2048.  The row size (4096) fits exactly in BLOCK_SIZE, so no
# masking is required.
#
# Constants:
#   N_ROWS = 8192   N_COLS = 4096   BLOCK_SIZE = 4096
#   ROWS_PER_PROG = 4  → grid = 2048
# ---------------------------------------------------------------------------

N_ROWS = 8192
N_COLS = 4096
BLOCK_SIZE = N_COLS          # full row per block
ROWS_PER_PROG = 4            # tuned for this shape
INV_N = 1.0 / N_COLS         # compile‑time reciprocal for faster division

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
    # Row range assigned to this program
    row_base = pid * ROWS_PER_PROG

    # Vectorised offsets for one complete row – no mask needed
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)

    for r in range(ROWS_PER_PROG):
        row_idx = row_base + r
        if row_idx >= n_rows:
            break

        row_start = row_idx * n_cols
        x_ptrs = x_ptr + row_start + offsets
        x = tl.load(x_ptrs, eviction_policy='evict_first').to(tl.float32)

        # Single‑pass sums
        s  = tl.sum(x, axis=0).to(tl.float32)
        sq = tl.sum(x * x, axis=0).to(tl.float32)

        # Mean and population variance using compile‑time inverse
        mean = s * INV_N
        var  = (sq * INV_N) - mean * mean

        # Store to output [2, n_rows]
        out_stride = n_rows
        tl.store(out_ptr + 0 * out_stride + row_idx, mean)
        tl.store(out_ptr + 1 * out_stride + row_idx, var)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    Compute per‑row mean and population variance of a 2‑D float32 tensor.

    Equivalent to:
        mean = x.mean(dim=-1)
        var  = x.var(dim=-1, unbiased=False)
        torch.stack([mean, var])

    Args:
        x: (8192, 4096) float32 CUDA tensor.
    Returns:
        out: (2, 8192) float32 tensor.  Row 0 = means, row 1 = variances.
    """
    assert x.is_cuda and x.dtype == torch.float32
    assert x.shape == (N_ROWS, N_COLS), \
        f"Expected shape ({N_ROWS}, {N_COLS}), got {x.shape}"

    out = torch.empty((2, N_ROWS), dtype=torch.float32, device=x.device)

    grid = (N_ROWS // ROWS_PER_PROG,)   # 2048 programs
    welford_kernel[grid](
        x,
        out,
        n_rows=N_ROWS,
        n_cols=N_COLS,
        BLOCK_SIZE=BLOCK_SIZE,
        ROWS_PER_PROG=ROWS_PER_PROG,
        INV_N=INV_N,
        num_warps=4,
        num_stages=4,       # low overhead, no pipelining needed
    )
    return out