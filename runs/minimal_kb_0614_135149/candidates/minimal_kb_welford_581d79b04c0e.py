import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel: per‑row mean and population variance (single‑pass sum/sumsq).
#
# Design:
#   - One program per row.  Each row is exactly 4096 float32 elements.
#   - The entire row fits in one block → no masking, fully coalesced loads.
#   - Single‑pass reduction: sum and sum‑of‑squares computed together.
#   - Tuning knobs: BLOCK_SIZE (must divide N_COLS), num_warps, num_stages,
#     and whether to use the compile‑time reciprocal INV_N.
#
# Expected for fixed shape (8192, 4096):
#   - N_ROWS  = 8192
#   - N_COLS  = 4096
#   - BLOCK_SIZE = 4096
#   - Configurable warps / stages (4 warps, 4 stages used as baseline).
# ---------------------------------------------------------------------------

N_ROWS = 8192
N_COLS = 4096
BLOCK_SIZE = N_COLS          # full row per block
INV_N = 1.0 / N_COLS

@triton.jit
def welford_kernel(
    x_ptr,
    out_ptr,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INV_N: tl.constexpr,
):
    pid = tl.program_id(0)              # row index
    row_start = pid * n_cols

    # Contiguous offsets for the whole row — no mask needed.
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
    x_ptrs = x_ptr + row_start + offsets

    # Load the row.  Cache hint: data is streamed once.
    x = tl.load(x_ptrs, eviction_policy='evict_first').to(tl.float32)

    # Single‑pass sums in float32.
    s  = tl.sum(x, axis=0).to(tl.float32)
    sq = tl.sum(x * x, axis=0).to(tl.float32)

    # Mean and population variance (unbiased=False).
    mean = s * INV_N
    var  = (sq * INV_N) - mean * mean

    # Output layout: [2, n_rows]   row0 = means, row1 = variances.
    out_stride = n_rows
    tl.store(out_ptr + 0 * out_stride + pid, mean)
    tl.store(out_ptr + 1 * out_stride + pid, var)


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

    grid = (N_ROWS,)
    welford_kernel[grid](
        x,
        out,
        n_rows=N_ROWS,
        n_cols=N_COLS,
        BLOCK_SIZE=BLOCK_SIZE,
        INV_N=INV_N,
        num_warps=4,      # tunable: 4 or 8
        num_stages=4,     # tunable: 2, 3, 4
    )
    return out