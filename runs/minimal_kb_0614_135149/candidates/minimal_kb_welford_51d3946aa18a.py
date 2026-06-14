import torch
import triton
import triton.language as tl

# Fixed dimensions for this operator
N_ROWS = 8192
N_COLS = 4096

@triton.jit
def welford_kernel(
    x_ptr,                 # [N_ROWS, N_COLS] float32 input
    out_ptr,               # [2, N_ROWS] float32 output (row0=mean, row1=var)
    n_rows: tl.constexpr,  # compile‑time constant: 8192
    n_cols: tl.constexpr,  # compile‑time constant: 4096
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)                     # row index
    row_start = pid * n_cols

    # Load the entire row without masking – n_cols divides BLOCK_SIZE exactly.
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
    x_ptrs = x_ptr + row_start + offsets
    x = tl.load(x_ptrs, eviction_policy='evict_first')

    # Single‑pass sum and sum of squares in fp32.
    s = tl.sum(x, axis=0).to(tl.float32)
    sq = tl.sum(x * x, axis=0).to(tl.float32)

    # Compute mean and population variance.
    mean = s / n_cols
    var = (sq / n_cols) - mean * mean

    # Store results: output layout is [2, n_rows].
    out_stride = n_rows
    tl.store(out_ptr + 0 * out_stride + pid, mean)
    tl.store(out_ptr + 1 * out_stride + pid, var)


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
    assert x.shape == (N_ROWS, N_COLS), f"Expected ({N_ROWS}, {N_COLS}) but got {x.shape}"

    out = torch.empty((2, N_ROWS), dtype=torch.float32, device=x.device)

    grid = (N_ROWS,)
    welford_kernel[grid](
        x,
        out,
        n_rows=N_ROWS,
        n_cols=N_COLS,
        BLOCK_SIZE=N_COLS,    # entire row fits in one block
        num_warps=4,          # good balance for memory‑bound reduction
        num_stages=4,         # enough stages for modest pipelining overhead
    )
    return out