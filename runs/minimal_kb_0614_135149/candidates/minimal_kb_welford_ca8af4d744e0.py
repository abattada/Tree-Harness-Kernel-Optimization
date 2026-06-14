import torch
import triton
import triton.language as tl

# -----------------------------------------------------------------------
# Clean, tunable Triton kernel for per‑row mean and population variance.
# Shape is fixed: (8192, 4096) – each row fits into one block (4096 elements).
# Tuning knobs: BLOCK_SIZE (must equal 4096 for whole row), num_warps,
# num_stages, and the use of tl.sum + tl.constexpr for the divisor.
# -----------------------------------------------------------------------

N_COLS = 4096
N_ROWS = 8192

@triton.jit
def welford_kernel(
    x_ptr,                         # pointer to input [N_ROWS, N_COLS]
    out_ptr,                       # pointer to output [2, N_ROWS]
    n_rows: tl.constexpr,          # 8192
    n_cols: tl.constexpr,          # 4096
    BLOCK_SIZE: tl.constexpr,      # = n_cols, full row in one block
):
    pid = tl.program_id(0)         # row index
    row_start = pid * n_cols

    # Offsets for the entire row – no mask required because BLOCK_SIZE == n_cols.
    offsets = tl.arange(0, BLOCK_SIZE)
    x_ptrs = x_ptr + row_start + offsets

    # Each row is read only once; evict_first keeps it out of cache.
    x = tl.load(x_ptrs, eviction_policy='evict_first')

    # Single‑pass sum and sum of squares in fp32.
    s  = tl.sum(x, axis=0).to(tl.float32)
    sq = tl.sum(x * x, axis=0).to(tl.float32)

    # Compute statistics using the compile‑time count to avoid division overhead.
    inv_n = 1.0 / n_cols
    mean = s * inv_n
    var = (sq * inv_n) - mean * mean   # population variance

    # Output: [2, n_rows] layout.
    out_stride = n_rows
    tl.store(out_ptr + 0 * out_stride + pid, mean)
    tl.store(out_ptr + 1 * out_stride + pid, var)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    Compute per‑row mean and population variance.
    Equivalent to:
        mean = x.mean(dim=-1)
        var  = x.var(dim=-1, unbiased=False)
        return torch.stack([mean, var])

    Input shape must be (8192, 4096), float32, CUDA.
    """
    assert x.is_cuda and x.dtype == torch.float32
    assert x.shape == (N_ROWS, N_COLS), f"Expected ({N_ROWS},{N_COLS}) got {x.shape}"

    out = torch.empty((2, N_ROWS), dtype=torch.float32, device=x.device)

    grid = (N_ROWS,)
    welford_kernel[grid](
        x,
        out,
        n_rows=N_ROWS,
        n_cols=N_COLS,
        BLOCK_SIZE=N_COLS,          # full row
        num_warps=4,                # tunable
        num_stages=2,               # tunable (no pipelining needed)
    )
    return out