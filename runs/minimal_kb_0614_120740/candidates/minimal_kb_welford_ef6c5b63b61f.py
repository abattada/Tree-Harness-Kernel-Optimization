import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel: per‑row mean and population variance via single‑pass sum/sumsq.
# Optimizations:
#   - All shape parameters are tl.constexpr, enabling aggressive compile‑time
#     folding and vectorization.
#   - tl.max_contiguous hints guarantee contiguous 128‑byte aligned accesses.
#   - eviction_policy='evict_first' since input is streamed once.
#   - num_stages=4 to improve memory‑level parallelism (pipelining).
#   - Precomputed reciprocal INV_N replaces division with multiplication.
# ---------------------------------------------------------------------------

N_ROWS = 8192
N_COLS = 4096
BLOCK_SIZE = N_COLS               # entire row fits in one block, no masking needed
INV_N = 1.0 / N_COLS              # compile‑time reciprocal

@triton.jit
def welford_kernel(
    x_ptr: tl.tensor,                 # [N_ROWS, N_COLS] input
    out_ptr: tl.tensor,               # [2, N_ROWS] output (row0=mean, row1=var)
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INV_N: tl.constexpr,
):
    pid = tl.program_id(0)            # row index
    row_start = pid * n_cols

    # Use tl.max_contiguous to hint the compiler about alignment and contiguity.
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
    x_ptrs = x_ptr + row_start + offsets

    # Load the entire row; input is used only once → evict_first.
    x = tl.load(x_ptrs, eviction_policy='evict_first')

    # Single‑pass sum and sum of squares in fp32.
    s  = tl.sum(x, axis=0).to(tl.float32)
    sq = tl.sum(x * x, axis=0).to(tl.float32)

    mean = s * INV_N
    var = (sq * INV_N) - mean * mean   # population variance

    # Store results – output layout: [2, n_rows]  (stride = n_rows)
    tl.store(out_ptr + 0 * n_rows + pid, mean)
    tl.store(out_ptr + 1 * n_rows + pid, var)


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
    assert x.shape == (N_ROWS, N_COLS), f"Expected ({N_ROWS}, {N_COLS}) but got {x.shape}"

    out = torch.empty(2, N_ROWS, dtype=torch.float32, device=x.device)

    grid = (N_ROWS,)                     # one program per row
    welford_kernel[grid](
        x,
        out,
        n_rows=N_ROWS,
        n_cols=N_COLS,
        BLOCK_SIZE=BLOCK_SIZE,
        INV_N=INV_N,
        num_warps=4,
        num_stages=4,                    # increased from 2 for better pipelining
    )
    return out