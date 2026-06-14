import torch
import triton
import triton.language as tl

# Fixed dimensions – no masking needed, full row per block
N_ROWS = 8192
N_COLS = 4096
BLOCK_SIZE = N_COLS
INV_N = 1.0 / N_COLS   # compile‑time reciprocal to avoid division at run time

@triton.jit
def welford_kernel(
    x_ptr,                  # input [N_ROWS, N_COLS] float32
    out_ptr,                # output [2, N_ROWS] float32
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INV_N: tl.constexpr,
):
    pid = tl.program_id(0)          # row index
    row_start = pid * n_cols

    # Load the whole row contiguously – n_cols divides BLOCK_SIZE exactly
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
    x_ptrs = x_ptr + row_start + offsets
    x = tl.load(x_ptrs, eviction_policy='evict_first')

    # Single‑pass sum / sum‑of‑squares (all in fp32)
    s  = tl.sum(x, axis=0).to(tl.float32)
    sq = tl.sum(x * x, axis=0).to(tl.float32)

    # Population mean and variance (use the pre‑computed inverse)
    mean = s * INV_N
    var = (sq * INV_N) - mean * mean

    # Store results into the [2, N_ROWS] output layout
    tl.store(out_ptr + 0 * n_rows + pid, mean)
    tl.store(out_ptr + 1 * n_rows + pid, var)


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
        x, out,
        n_rows=N_ROWS,
        n_cols=N_COLS,
        BLOCK_SIZE=BLOCK_SIZE,
        INV_N=INV_N,
        num_warps=8,          # more warps → better memory parallelism for this bandwidth‑bound reduction
        num_stages=2,         # low stages to keep register pressure low (no pipelining needed)
    )
    return out