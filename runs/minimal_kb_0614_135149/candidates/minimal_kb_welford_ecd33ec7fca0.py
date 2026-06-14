import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel: per‑row mean and population variance (single‑pass sum/sumsq).
#
# This version refines the parent by:
#   • Adding alignment hints (tl.multiple_of on the base pointer) together
#     with tl.max_contiguous to let the compiler emit wider vectorized loads.
#   • Keeping the compile‑time constant INV_N for fused multiply of reciprocal.
#   • Retaining eviction_policy='evict_first' because the input is streamed
#     exactly once.
#
# Expected for fixed shape (8192, 4096):
#   - N_ROWS  = 8192
#   - N_COLS  = 4096
#   - BLOCK_SIZE = N_COLS
#   - Each program processes exactly one full row with no masking.
# ---------------------------------------------------------------------------

N_ROWS = 8192
N_COLS = 4096
BLOCK_SIZE = N_COLS          # entire row per block
INV_N = 1.0 / N_COLS         # compile‑time reciprocal

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

    # Offsets for the whole row – no mask.
    # Hints: the offsets are max_contiguous and aligned, enabling wide loads.
    offsets = tl.max_contiguous(
        tl.multiple_of(tl.arange(0, BLOCK_SIZE), 16),  # 16‑element alignment
        BLOCK_SIZE,
    )
    # Base pointer alignment hint: the underlying data is 128‑byte aligned.
    x_base = tl.multiple_of(x_ptr, 128)
    x_ptrs = x_base + row_start + offsets

    # Load the row with streaming eviction policy.
    x = tl.load(x_ptrs, eviction_policy='evict_first').to(tl.float32)

    # Single‑pass sum and sum of squares.
    s  = tl.sum(x, axis=0).to(tl.float32)
    sq = tl.sum(x * x, axis=0).to(tl.float32)

    # Mean and population variance (unbiased=False).
    mean = s * INV_N
    var  = (sq * INV_N) - mean * mean

    # Output layout: [2, n_rows] — row 0 = means, row 1 = variances.
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

    Input:  x shape (8192, 4096), float32, CUDA, contiguous.
    Output: out shape (2, 8192), float32.
    """
    assert x.is_cuda and x.dtype == torch.float32
    assert x.is_contiguous()
    assert x.shape == (N_ROWS, N_COLS), \
        f"Expected shape ({N_ROWS}, {N_COLS}), got {x.shape}"
    # Ensure base alignment for the vectorization hint to be valid.
    assert x.data_ptr() % 128 == 0, "Input data must be 128‑byte aligned"

    out = torch.empty((2, N_ROWS), dtype=torch.float32, device=x.device)

    grid = (N_ROWS,)
    welford_kernel[grid](
        x,
        out,
        n_rows=N_ROWS,
        n_cols=N_COLS,
        BLOCK_SIZE=BLOCK_SIZE,
        INV_N=INV_N,
        num_warps=4,      # good balance for this memory‑bound kernel
        num_stages=4,     # moderate pipelining; does not increase register pressure
    )
    return out