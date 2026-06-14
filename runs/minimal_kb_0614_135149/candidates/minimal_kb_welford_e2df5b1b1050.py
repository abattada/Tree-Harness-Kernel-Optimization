import torch
import triton
import triton.language as tl

# Fixed dimensions; compile‑time constants inform the compiler aggressively.
N_ROWS = 8192
N_COLS = 4096
BLOCK_SIZE = N_COLS
INV_N = 1.0 / N_COLS          # precomputed reciprocal – faster than division

@triton.jit
def welford_kernel(
    x_ptr,                      # [N_ROWS, N_COLS] float32 input
    out_ptr,                    # [2, N_ROWS] output (row 0 = mean, row 1 = var)
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INV_N: tl.constexpr,
):
    pid = tl.program_id(0)      # row index
    row_start = pid * n_cols

    # Load the entire row without masking – BLOCK_SIZE == N_COLS.
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
    x_ptrs = x_ptr + row_start + offsets
    x = tl.load(x_ptrs, eviction_policy='evict_first')

    # Single‑pass sum and sum of squares.
    s  = tl.sum(x, axis=0).to(tl.float32)
    sq = tl.sum(x * x, axis=0).to(tl.float32)

    # Use the precomputed reciprocal for cheaper arithmetic.
    mean = s * INV_N
    var  = (sq * INV_N) - mean * mean   # population variance

    # Output layout: [2, n_rows]
    tl.store(out_ptr + 0 * n_rows + pid, mean)
    tl.store(out_ptr + 1 * n_rows + pid, var)

def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    Compute per‑row mean and population variance of a (8192, 4096) float32 tensor.
    Returns a tensor of shape (2, 8192) where row 0 contains means and row 1
    contains the corresponding population variances.
    """
    assert x.is_cuda and x.dtype == torch.float32
    assert x.shape == (N_ROWS, N_COLS), f"Expected ({N_ROWS}, {N_COLS}) but got {x.shape}"

    out = torch.empty((2, N_ROWS), dtype=torch.float32, device=x.device)

    grid = (N_ROWS,)
    welford_kernel[grid](
        x, out,
        n_rows=N_ROWS,
        n_cols=N_COLS,
        BLOCK_SIZE=BLOCK_SIZE,
        INV_N=INV_N,
        num_warps=4,
        num_stages=4,
    )
    return out