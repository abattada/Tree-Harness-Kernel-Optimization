import torch
import triton
import triton.language as tl

# Fixed dimensions – specialized as compile‑time constants
N_ROWS = 8192
N_COLS = 4096
BLOCK_SIZE = N_COLS          # entire row fits in one block
INV_N = 1.0 / N_COLS         # reciprocal for faster multiply instead of divide

@triton.jit
def welford_kernel(
    x_ptr,                    # [N_ROWS, N_COLS] float32 input
    out_ptr,                  # [2, N_ROWS] float32 output (row0=mean, row1=var)
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INV_N: tl.constexpr,
):
    pid = tl.program_id(0)                      # row index
    # Hint alignment: each row starts at a 128‑byte aligned address
    row_start = tl.multiple_of(pid * n_cols, 128)

    # Vectorized, coalesced load of the whole row with eviction policy
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
    x = tl.load(x_ptr + row_start + offsets, eviction_policy='evict_first')

    # Single‑pass sum and sum of squares in float32
    s  = tl.sum(x, axis=0).to(tl.float32)
    sq = tl.sum(x * x, axis=0).to(tl.float32)

    # Compute mean and population variance using fast multiplication
    mean = s * INV_N
    var = (sq * INV_N) - mean * mean

    # Store results – output layout is [2, n_rows]
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
        BLOCK_SIZE=BLOCK_SIZE,
        INV_N=INV_N,
        num_warps=4,          # good balance for memory‑bound reduction
        num_stages=2,         # no pipelining needed; lower pressure
    )
    return out