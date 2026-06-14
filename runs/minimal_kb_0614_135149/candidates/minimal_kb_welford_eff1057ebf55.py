import torch
import triton
import triton.language as tl

# Fixed dimensions
N_ROWS = 8192
N_COLS = 4096
BLOCK_SIZE = N_COLS          # entire row fits in a single block – no masking
INV_N = 1.0 / N_COLS         # compile-time reciprocal for fast division
ROWS_PER_PROG = 16           # each program processes this many rows
NUM_WARPS = 4
NUM_STAGES = 4

@triton.jit
def welford_kernel(
    x_ptr,                    # [N_ROWS, N_COLS] float32 input
    out_ptr,                  # [2, N_ROWS] float32 output (row0=mean, row1=var)
    n_rows: tl.constexpr,     # 8192
    n_cols: tl.constexpr,     # 4096
    BLOCK_SIZE: tl.constexpr, # 4096
    INV_N: tl.constexpr,      # 1.0 / 4096
    ROWS_PER_PROG: tl.constexpr,  # number of rows processed per program
):
    pid = tl.program_id(0)
    start_row = pid * ROWS_PER_PROG  # first row assigned to this program

    # Offsets for a whole row (no mask needed – n_cols divides BLOCK_SIZE)
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)

    for r_off in range(ROWS_PER_PROG):
        row = start_row + r_off
        if row < n_rows:
            row_start = row * n_cols
            x_ptrs = x_ptr + row_start + offsets
            # Load entire row – evict_first because each row is read exactly once
            x = tl.load(x_ptrs, eviction_policy='evict_first')

            # Single-pass sum and sum of squares in fp32
            s = tl.sum(x, axis=0).to(tl.float32)
            sq = tl.sum(x * x, axis=0).to(tl.float32)

            # Compute population statistics using compile-time inverse
            mean = s * INV_N
            var = (sq * INV_N) - mean * mean

            # Store into output [2, n_rows]
            out_stride = n_rows
            tl.store(out_ptr + 0 * out_stride + row, mean)
            tl.store(out_ptr + 1 * out_stride + row, var)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    Compute per-row mean and population variance of a float32 tensor.

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

    # Grid: each program handles up to ROWS_PER_PROG consecutive rows
    grid = (N_ROWS + ROWS_PER_PROG - 1) // ROWS_PER_PROG

    welford_kernel[grid](
        x, out,
        n_rows=N_ROWS,
        n_cols=N_COLS,
        BLOCK_SIZE=BLOCK_SIZE,
        INV_N=INV_N,
        ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return out