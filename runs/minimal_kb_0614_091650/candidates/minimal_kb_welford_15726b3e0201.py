import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel: per‑row mean and population variance via single‑pass sum/sumsq.
# This version improves on the parent by:
#   • Making n_rows/n_cols tl.constexpr so the compiler can fold constants and
#     enable more aggressive optimizations.
#   • Adding eviction_policy='evict_first' on the input loads, since each row
#     is read exactly once and should not pollute L2/L1 for subsequent rows.
#   • Using default num_warps=4 (which is often optimal for memory‑bound kernels)
#     and num_stages=2 (pipelining not needed here, but low stages reduce
#     register pressure).
# ---------------------------------------------------------------------------

@triton.jit
def welford_kernel(
    x_ptr,                    # [n_rows, n_cols] contiguous input
    out_ptr,                  # [2, n_rows] output (row0=mean, row1=var)
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)                 # row index
    row_start = pid * n_cols

    # Offsets for the whole row – no mask needed because n_cols divides BLOCK_SIZE.
    offsets = tl.arange(0, BLOCK_SIZE)
    x_ptrs = x_ptr + row_start + offsets

    # Load the row with eviction hint: input data is only used once.
    x = tl.load(x_ptrs, eviction_policy='evict_first')

    # Single‑pass sum and sum of squares in fp32.
    s = tl.sum(x, axis=0).to(tl.float32)
    sq = tl.sum(x * x, axis=0).to(tl.float32)

    mean = s / n_cols
    var = (sq / n_cols) - mean * mean   # population variance

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
    assert x.is_cuda and x.dtype == torch.float32
    n_rows, n_cols = x.shape
    # Must be exactly this shape to avoid masking (full row per block).
    assert n_cols == 4096, f"Expected n_cols=4096, got {n_cols}"

    out = torch.empty((2, n_rows), dtype=torch.float32, device=x.device)

    grid = (n_rows,)
    welford_kernel[grid](
        x,
        out,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE=n_cols,          # 4096 – single block covers entire row
        num_warps=4,
        num_stages=2,               # no pipelining needed; keep stages low
    )
    return out