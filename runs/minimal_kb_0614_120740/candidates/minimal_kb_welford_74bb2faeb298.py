import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel: compute per‑row mean and population variance via single‑pass
# sum and sum of squares.  Each program handles one entire row (4096 columns).
# We exploit the fixed shape for compile‑time constants and to avoid masking.
# ---------------------------------------------------------------------------

@triton.jit
def welford_kernel(
    x_ptr,                    # [n_rows, n_cols] contiguous input
    out_ptr,                  # [2, n_rows] output (row0 = mean, row1 = var)
    n_rows: tl.constexpr,     # 8192
    n_cols: tl.constexpr,     # 4096
    BLOCK_SIZE: tl.constexpr, # 4096 (must equal n_cols)
    INV_N: tl.constexpr,      # 1.0 / n_cols
):
    pid = tl.program_id(0)                 # row index
    row_start = pid * n_cols

    # Offsets for the whole row – no mask needed because n_cols divides BLOCK_SIZE.
    offsets = tl.arange(0, BLOCK_SIZE)
    x_ptrs = x_ptr + row_start + offsets

    # Load the row with eviction hint: input data is used only once.
    x = tl.load(x_ptrs, eviction_policy='evict_first')

    # Single‑pass sum and sum of squares in fp32.
    s = tl.sum(x, axis=0).to(tl.float32)
    sq = tl.sum(x * x, axis=0).to(tl.float32)

    # Compute mean and population variance using compile‑time reciprocal.
    mean = s * INV_N
    # var = (sq / n_cols) - mean^2
    var = (sq * INV_N) - mean * mean

    # Store results – output layout: [2, n_rows]
    out_stride = n_rows
    tl.store(out_ptr + 0 * out_stride + pid, mean)
    tl.store(out_ptr + 1 * out_stride + pid, var)


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
    assert x.is_cuda and x.dtype == torch.float32, "x must be CUDA float32"
    n_rows, n_cols = x.shape
    assert n_cols == 4096, f"Expected n_cols=4096, got {n_cols}"  # fixed for this operator

    out = torch.empty((2, n_rows), dtype=torch.float32, device=x.device)

    # Compile‑time constants
    BLOCK_SIZE = 4096
    INV_N = 1.0 / 4096

    grid = (n_rows,)
    welford_kernel[grid](
        x,
        out,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        INV_N=INV_N,
        num_warps=4,
        num_stages=2,
    )
    return out