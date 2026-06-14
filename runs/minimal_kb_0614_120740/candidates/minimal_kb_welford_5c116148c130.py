import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel: per‑row mean and population variance via single‑pass sum/sumsq.
# The shape is fixed to (8192, 4096) to fully eliminate masking and allow
# compile‑time optimisations.  Each program processes exactly one row.
# This version adds tl.max_contiguous hint for better vectorized loads.
# ---------------------------------------------------------------------------

@triton.jit
def welford_kernel(
    x_ptr: tl.tensor,                     # [n_rows, n_cols] input
    out_ptr: tl.tensor,                   # [2, n_rows] output (row0=mean, row1=var)
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INV_N: tl.constexpr,                  # 1.0 / n_cols, computed at compile time
):
    pid = tl.program_id(0)                # row index
    row_start = pid * n_cols

    # contiguity hint helps the compiler generate wide vectorized loads
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
    x_ptrs = x_ptr + row_start + offsets

    # Load the entire row; evict_first because this data is used only once.
    x = tl.load(x_ptrs, eviction_policy='evict_first')

    # Single‑pass sum and sum of squares in fp32.
    s  = tl.sum(x, axis=0).to(tl.float32)
    sq = tl.sum(x * x, axis=0).to(tl.float32)

    mean = s * INV_N
    var = (sq * INV_N) - mean * mean      # population variance

    # Output layout: [2, n_rows]
    out_stride = n_rows
    tl.store(out_ptr + 0 * out_stride + pid, mean)
    tl.store(out_ptr + 1 * out_stride + pid, var)


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
    assert x.shape == (n_rows, n_cols), f"Expected ({n_rows}, {n_cols}) but got {x.shape}"

    out = torch.empty((2, n_rows), dtype=torch.float32, device=x.device)

    grid = (n_rows,)
    welford_kernel[grid](
        x,
        out,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE=n_cols,       # 4096, one block covers a whole row
        INV_N=1.0 / n_cols,      # compile‑time reciprocal
        num_warps=4,
        num_stages=2,            # pipelining not needed; low stages reduce register pressure
    )
    return out