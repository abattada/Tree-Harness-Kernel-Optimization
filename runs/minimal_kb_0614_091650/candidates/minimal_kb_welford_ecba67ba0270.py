import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel: compute per‑row mean and population (unbiased=False) variance
# for a 2‑D float32 tensor. Uses constexpr for compile‑time specialization.
# ---------------------------------------------------------------------------
# Tunable parameters (kept from parent version for optimal occupancy)
NUM_WARPS   = 8
NUM_STAGES  = 4
BLOCK_SIZE  = 4096   # must equal n_cols for this fixed shape

@triton.jit
def welford_kernel(
    x_ptr,                         # [n_rows, n_cols] input (contiguous)
    out_ptr,                       # [2, n_rows] output (row0=mean, row1=var)
    n_rows: tl.constexpr,          # row count -> constexpr for stride hints
    n_cols: tl.constexpr,          # col count -> constexpr for vectorization
    BLOCK_SIZE: tl.constexpr,      # tile size (== n_cols in our use)
):
    pid = tl.program_id(0)          # row index
    row_start = pid * n_cols

    # Load one full row – no mask needed because BLOCK_SIZE == n_cols
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + row_start + offsets)

    # Single‑pass sum and sum of squares (float32)
    s  = tl.sum(x, axis=0)
    sq = tl.sum(x * x, axis=0)

    # Population mean and variance
    mean = s / n_cols
    var  = (sq / n_cols) - mean * mean

    # Store results – output layout: [2, n_rows]
    tl.store(out_ptr + 0 * n_rows + pid, mean)
    tl.store(out_ptr + 1 * n_rows + pid, var)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    Equivalent to:
        mean = x.mean(dim=-1)
        var  = x.var(dim=-1, unbiased=False)
        return torch.stack([mean, var])

    Input:  x shape (8192, 4096), dtype float32, CUDA device
    Output: out shape (2, 8192), dtype float32
    """
    assert x.is_cuda and x.dtype == torch.float32
    n_rows, n_cols = x.shape
    assert n_cols == 4096, "This kernel assumes n_cols = 4096 for full‑row tiles"

    out = torch.empty((2, n_rows), dtype=torch.float32, device=x.device)

    grid = (n_rows,)
    welford_kernel[grid](
        x, out,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return out