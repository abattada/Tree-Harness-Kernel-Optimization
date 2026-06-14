import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel: per‑row mean and population variance via single‑pass sum/sumsq.
# Each program handles ROWS_PER_PROG rows (8) to reduce launch overhead and
# improve L2 reuse for output writes.
# ---------------------------------------------------------------------------

@triton.jit
def welford_kernel(
    x_ptr,                     # [n_rows, n_cols] contiguous input
    out_ptr,                   # [2, n_rows] output (row0=mean, row1=var)
    n_rows: tl.constexpr,      # 8192
    n_cols: tl.constexpr,      # 4096
    BLOCK_SIZE: tl.constexpr,  # must equal n_cols
    ROWS_PER_PROG: tl.constexpr, # 8
):
    # Each program processes a chunk of consecutive rows.
    block_id = tl.program_id(0)
    row_start = block_id * ROWS_PER_PROG

    # Offsets covering an entire row – no mask needed because n_cols exactly divides BLOCK_SIZE.
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)

    # Loop over rows in the chunk
    for i in range(ROWS_PER_PROG):
        row = row_start + i
        # Load the entire row with eviction hint: input is read exactly once.
        x_ptrs = x_ptr + row * n_cols + offsets
        x = tl.load(x_ptrs, eviction_policy='evict_first')

        # Single-pass sum and sum of squares in fp32.
        s  = tl.sum(x, axis=0).to(tl.float32)
        sq = tl.sum(x * x, axis=0).to(tl.float32)

        mean = s / n_cols
        var  = (sq / n_cols) - mean * mean   # population variance

        # Store results – output layout: [2, n_rows]
        tl.store(out_ptr + 0 * n_rows + row, mean)
        tl.store(out_ptr + 1 * n_rows + row, var)


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
    n_rows, n_cols = x.shape
    # Must be exactly 8192 x 4096 to avoid masking.
    assert n_cols == 4096, f"Expected n_cols=4096, got {n_cols}"

    out = torch.empty((2, n_rows), dtype=torch.float32, device=x.device)

    # Configuration
    ROWS_PER_PROG = 8
    grid = (n_rows // ROWS_PER_PROG,)
    welford_kernel[grid](
        x,
        out,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE=n_cols,
        ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=4,
        num_stages=4,
    )
    return out