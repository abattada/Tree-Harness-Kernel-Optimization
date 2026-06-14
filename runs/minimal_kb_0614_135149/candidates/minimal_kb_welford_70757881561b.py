import torch
import triton
import triton.language as tl

# Fixed dimensions for the target problem – no dynamic shape handling needed.
N_ROWS: int = 8192
N_COLS: int = 4096
BLOCK_SIZE: int = N_COLS          # a whole row fits in one block
INV_N: float = 1.0 / N_COLS      # compile-time reciprocal to avoid runtime divisions

# Grid‑stride loop: each program processes several rows to amortise launch overhead
# and keep the SM busy over multiple memory loads.
ROWS_PER_PROG: int = 4

@triton.jit
def welford_kernel(
    x_ptr,
    out_ptr,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INV_N: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    base_row = pid * ROWS_PER_PROG

    # Offsets for a whole contiguous row – the block is the exact row length.
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)

    # Process ROWS_PER_PROG rows sequentially inside the same block.
    for r in tl.static_range(ROWS_PER_PROG):
        row = base_row + r
        # Because N_ROWS is a multiple of ROWS_PER_PROG, no boundary mask is needed.
        row_start = row * n_cols
        x_ptrs = x_ptr + row_start + offsets
        x = tl.load(x_ptrs, eviction_policy='evict_first')

        # Single‑pass sum and sum‑of‑squares (all in float32).
        s  = tl.sum(x, axis=0).to(tl.float32)
        sq = tl.sum(x * x, axis=0).to(tl.float32)

        mean = s * INV_N
        var = (sq * INV_N) - mean * mean   # population variance

        # Store results into the [2, N_ROWS] output layout.
        tl.store(out_ptr + 0 * n_rows + row, mean)
        tl.store(out_ptr + 1 * n_rows + row, var)


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

    # Launch fewer programs – each one handles ROWS_PER_PROG rows.
    grid = (N_ROWS // ROWS_PER_PROG,)
    welford_kernel[grid](
        x, out,
        n_rows=N_ROWS,
        n_cols=N_COLS,
        BLOCK_SIZE=BLOCK_SIZE,
        INV_N=INV_N,
        ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=8,
        num_stages=2,
    )
    return out