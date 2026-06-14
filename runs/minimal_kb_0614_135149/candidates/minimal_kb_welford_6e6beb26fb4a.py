import torch
import triton
import triton.language as tl

# Fixed dimensions – no masking needed, full row per block
N_ROWS = 8192
N_COLS = 4096
BLOCK_SIZE = N_COLS  # each row fits in a single block


@triton.jit
def welford_kernel(
    x_ptr,                     # input  [N_ROWS, N_COLS] float32
    out_ptr,                   # output [2, N_ROWS]     float32
    n_rows: tl.constexpr,      # 8192
    n_cols: tl.constexpr,      # 4096
    BLOCK_SIZE: tl.constexpr,  # 4096
):
    pid = tl.program_id(0)          # row index
    row_start = pid * n_cols

    # Load the entire row with contiguous access and eviction hint
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
    x_ptrs = x_ptr + row_start + offsets
    x = tl.load(x_ptrs, eviction_policy='evict_first')

    # Cast to float64 before computing sums to avoid catastrophic cancellation
    # when variance is small relative to the mean.
    x64 = x.to(tl.float64)

    # Single‑pass sum and sum of squares in float64
    s  = tl.sum(x64, axis=0)
    sq = tl.sum(x64 * x64, axis=0)

    # Population mean and variance, computed in float64 then cast to float32
    mean = s / n_cols
    var = (sq / n_cols) - mean * mean

    # Store results – output layout: [2, n_rows]
    out_stride = n_rows
    tl.store(out_ptr + 0 * out_stride + pid, mean.to(tl.float32))
    tl.store(out_ptr + 1 * out_stride + pid, var.to(tl.float32))


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    Compute per‑row mean and population variance of a float32 tensor.

    Equivalent to:
        mean = x.mean(dim=-1)
        var  = x.var(dim=-1, unbiased=False)
        return torch.stack([mean, var])

    Args:
        x: torch.Tensor of shape (8192, 4096), dtype float32, on CUDA.
    Returns:
        Out tensor of shape (2, 8192), dtype float32.
    """
    assert x.is_cuda and x.dtype == torch.float32
    assert x.shape == (N_ROWS, N_COLS), \
        f"Expected ({N_ROWS}, {N_COLS}) but got {x.shape}"

    out = torch.empty((2, N_ROWS), dtype=torch.float32, device=x.device)

    grid = (N_ROWS,)
    welford_kernel[grid](
        x, out,
        n_rows=N_ROWS,
        n_cols=N_COLS,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,           # few warps are enough for this memory‑bound reduction
        num_stages=2,          # minimal stages to keep register pressure low
    )
    return out