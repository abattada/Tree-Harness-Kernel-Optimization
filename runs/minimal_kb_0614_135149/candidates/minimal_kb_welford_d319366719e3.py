import torch
import triton
import triton.language as tl

# Fixed shape for this operator: (8192, 4096)
N_ROWS = 8192
N_COLS = 4096
BLOCK_SIZE = N_COLS          # full row per block – no masking needed

@triton.jit
def welford_kernel(
    x_ptr,                    # pointer to input [N_ROWS, N_COLS]
    out_ptr,                  # pointer to output [2, N_ROWS] (row0=mean, row1=var)
    n_rows: tl.constexpr,     # 8192
    n_cols: tl.constexpr,     # 4096
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)                 # row index
    row_start = pid * n_cols

    # Offsets for the whole contiguous row – no mask required.
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
    x_ptrs = x_ptr + row_start + offsets

    # Load the row once with eviction hint "evict_first".
    x = tl.load(x_ptrs, eviction_policy='evict_first')

    # Single-pass sum and sum of squares in fp32.
    s  = tl.sum(x, axis=0).to(tl.float32)
    sq = tl.sum(x * x, axis=0).to(tl.float32)

    # Mean and population variance (unbiased=False).
    # Division by n_cols is hoisted via constant folding.
    mean = s / n_cols
    var  = (sq / n_cols) - mean * mean

    # Output layout: [2, n_rows] – row 0 = mean, row 1 = variance.
    out_stride = n_rows
    tl.store(out_ptr + 0 * out_stride + pid, mean)
    tl.store(out_ptr + 1 * out_stride + pid, var)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    Compute per‑row mean and population variance for a (8192,4096) float32 tensor.

    Equivalent to:
        mean = x.mean(dim=-1)
        var  = x.var(dim=-1, unbiased=False)
        torch.stack([mean, var])

    Returns:
        out: (2, 8192) float32 tensor on the same CUDA device.
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
        # Tuned lauch parameters for memory-bound single-row reduction:
        # 8 warps increase memory-level parallelism, 2 stages reduce register pressure.
        num_warps=8,
        num_stages=2,
    )
    return out