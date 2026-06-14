import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel: per‑row mean and population variance via single‑pass sum/sumsq.
# This version adds tl.max_contiguous to help the compiler vectorize loads,
# and uses num_stages=4 (common in the reference) while keeping the efficient
# compile‑time reciprocal multiplication.
# ---------------------------------------------------------------------------

@triton.jit
def welford_kernel(
    x_ptr,                  # [n_rows, n_cols] input
    out_ptr,                # [2, n_rows] output (row0=mean, row1=var)
    n_rows: tl.constexpr,   # 8192
    n_cols: tl.constexpr,   # 4096
    BLOCK_SIZE: tl.constexpr,
    INV_N: tl.constexpr,    # 1.0 / n_cols
):
    pid = tl.program_id(0)          # row index
    row_start = pid * n_cols

    # Use tl.max_contiguous to hint the compiler for aligned vectorized loads.
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
    x_ptrs = x_ptr + row_start + offsets
    # evict_first because each row is read only once.
    x = tl.load(x_ptrs, eviction_policy='evict_first')

    # Single‑pass sum and sum of squares in fp32.
    s  = tl.sum(x, axis=0).to(tl.float32)
    sq = tl.sum(x * x, axis=0).to(tl.float32)

    # Compute mean and population variance using compile‑time reciprocal.
    mean = s * INV_N
    var = (sq * INV_N) - mean * mean

    # Store to output [2, n_rows]
    tl.store(out_ptr + 0 * n_rows + pid, mean)
    tl.store(out_ptr + 1 * n_rows + pid, var)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    Compute per‑row mean and population variance of a 2‑D float32 tensor.

    Reference:  mean = x.mean(dim=-1); var = x.var(dim=-1, unbiased=False)
                return torch.stack([mean, var])

    Input:  x shape (8192, 4096), float32, CUDA device.
    Output: out shape (2, 8192), float32.
    """
    assert x.is_cuda and x.dtype == torch.float32
    n_rows, n_cols = 8192, 4096
    # Fixed shape – no masking needed.
    assert x.shape == (n_rows, n_cols), f"Expected ({n_rows}, {n_cols}) but got {x.shape}"

    out = torch.empty((2, n_rows), dtype=torch.float32, device=x.device)

    grid = (n_rows,)
    welford_kernel[grid](
        x,
        out,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE=n_cols,        # 4096 covers the whole row
        INV_N=1.0 / n_cols,       # compile‑time reciprocal
        num_warps=4,
        num_stages=4,             # slightly more stages may help pipeline loads
    )
    return out