import torch
import triton
import triton.language as tl

# Fixed dimensions for this operator
N_ROWS = 8192
N_COLS = 4096
BLOCK_SIZE = N_COLS          # entire row fits in one block
ROWS_PER_PROG = 8            # process 8 rows per program
INV_N = 1.0 / float(N_COLS)  # precomputed reciprocal


@triton.jit
def welford_kernel(
    x_ptr,
    out_ptr,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
    INV_N: tl.constexpr,
):
    """
    Each program processes ROWS_PER_PROG consecutive rows.
    For each row: load, single-pass sum + sum-of-squares, then compute
    mean and population variance.  Results are stored to out[2, n_rows].
    """
    pid = tl.program_id(0)
    base_row = pid * ROWS_PER_PROG

    # Offsets cover the whole row – BLOCK_SIZE exactly divides n_cols
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)

    # Unrolled loop over the rows assigned to this program
    for r in range(ROWS_PER_PROG):
        row = base_row + r
        row_start = row * n_cols

        # Load the full row; evict_first avoids polluting caches for this
        # streaming read-once data.
        x_ptrs = x_ptr + row_start + offsets
        x = tl.load(x_ptrs, eviction_policy='evict_first')

        # Single-pass reduction in fp32
        s = tl.sum(x, axis=0).to(tl.float32)
        sq = tl.sum(x * x, axis=0).to(tl.float32)

        # Population mean and variance using compile-time inverse
        mean = s * INV_N
        var = sq * INV_N - mean * mean

        # Write to output: layout [2, n_rows]
        tl.store(out_ptr + 0 * n_rows + row, mean)
        tl.store(out_ptr + 1 * n_rows + row, var)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    Compute per-row mean and population variance of a float32 tensor.

    Equivalent to:
        mean = x.mean(dim=-1)
        var  = x.var(dim=-1, unbiased=False)
        return torch.stack([mean, var])

    Input:  x shape (8192, 4096), float32, CUDA device.
    Output: out shape (2, 8192), float32.
    """
    assert x.is_cuda and x.dtype == torch.float32
    assert x.shape == (N_ROWS, N_COLS), f"Expected ({N_ROWS}, {N_COLS}) but got {x.shape}"

    out = torch.empty((2, N_ROWS), dtype=torch.float32, device=x.device)

    # Launch one program per chunk of ROWS_PER_PROG rows
    grid = (N_ROWS // ROWS_PER_PROG,)
    welford_kernel[grid](
        x, out,
        n_rows=N_ROWS,
        n_cols=N_COLS,
        BLOCK_SIZE=BLOCK_SIZE,
        ROWS_PER_PROG=ROWS_PER_PROG,
        INV_N=INV_N,
        num_warps=4,
        num_stages=4,
    )
    return out