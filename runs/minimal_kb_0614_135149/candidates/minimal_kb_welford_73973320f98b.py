import torch
import triton
import triton.language as tl

# Shape is fixed for this operator – used as compile-time constants everywhere.
N_ROWS = 8192
N_COLS = 4096

@triton.jit
def welford_persistent_kernel(
    x_ptr,                  # [N_ROWS, N_COLS] input
    out_ptr,                # [2, N_ROWS] output (row0=mean, row1=var)
    counter_ptr,            # scalar int32 counter (initialised to 0)
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INV_N: tl.constexpr,    # = 1 / n_cols
):
    # Each persistent program atomically claims a row id.
    row_id = tl.atomic_add(counter_ptr, 1)
    while row_id < n_rows:
        row_start = row_id * n_cols
        offsets = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
        x = tl.load(x_ptr + row_start + offsets, eviction_policy='evict_first')

        # Single-pass sum and sum-of-squares.
        s  = tl.sum(x, axis=0).to(tl.float32)
        sq = tl.sum(x * x, axis=0).to(tl.float32)

        # Compute mean and population variance using compile-time reciprocal.
        mean = s * INV_N
        var  = (sq * INV_N) - mean * mean

        # Store results – output layout: [2, n_rows]
        tl.store(out_ptr + 0 * n_rows + row_id, mean)
        tl.store(out_ptr + 1 * n_rows + row_id, var)

        row_id = tl.atomic_add(counter_ptr, 1)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    Compute per-row mean and population variance of a float32 tensor.

    Input:  x shape (8192, 4096), dtype float32, CUDA.
    Output: tensor of shape (2, 8192) where row 0 = means, row 1 = vars.
    """
    assert x.is_cuda and x.dtype == torch.float32
    assert x.shape == (N_ROWS, N_COLS), f"Expected ({N_ROWS}, {N_COLS}), got {x.shape}"

    out = torch.empty((2, N_ROWS), dtype=torch.float32, device=x.device)
    counter = torch.zeros(1, dtype=torch.int32, device=x.device)

    INV_N = 1.0 / N_COLS

    # Launch a small grid of persistent programs. Each program processes many rows.
    grid = (256,)
    welford_persistent_kernel[grid](
        x, out, counter,
        n_rows=N_ROWS,
        n_cols=N_COLS,
        BLOCK_SIZE=N_COLS,          # entire row fits in one block
        INV_N=INV_N,
        num_warps=4,
        num_stages=2,               # kept low to minimise register pressure
    )
    return out