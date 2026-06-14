import torch
import triton
import triton.language as tl


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    Compute per-row population mean and variance using a multirow-per-program
    single-pass sum-of-squares approach.

    Args:
        x: float32 tensor of shape (8192, 4096)

    Returns:
        A float32 tensor of shape (2, 8192) where out[0, i] = mean of row i
        and out[1, i] = population variance of row i.
    """
    N_ROWS, N_COLS = x.shape
    out = torch.empty(2, N_ROWS, dtype=x.dtype, device=x.device)

    # Tuned for RTX 5090: each program processes ROWS_PER_PROG rows
    BLOCK_SIZE = 1024         # divides N_COLS (4096)
    ROWS_PER_PROG = 4         # must divide N_ROWS for mask-free execution
    NUM_WARPS = 8
    NUM_STAGES = 2

    assert N_COLS % BLOCK_SIZE == 0, "BLOCK_SIZE must divide N_COLS"
    assert N_ROWS % ROWS_PER_PROG == 0, "ROWS_PER_PROG must divide N_ROWS for full-tile path"

    grid = (N_ROWS // ROWS_PER_PROG,)
    welford_multirow_kernel[grid](
        x, out,
        x.stride(0),
        N_ROWS=N_ROWS, N_COLS=N_COLS,
        ROWS_PER_PROG=ROWS_PER_PROG,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return out


@triton.jit
def welford_multirow_kernel(x_ptr, out_ptr, stride_row,
                            N_ROWS: tl.constexpr, N_COLS: tl.constexpr,
                            ROWS_PER_PROG: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    Each thread block processes ROWS_PER_PROG rows sequentially.
    Per-row accumulation uses a vectorized tile loop over columns.
    """
    pid = tl.program_id(0)
    start_row = pid * ROWS_PER_PROG

    N_f = tl.cast(N_COLS, tl.float32)

    # Unrolled over the rows handled by this program
    for i in tl.static_range(0, ROWS_PER_PROG):
        r = start_row + i                      # row index (no mask needed)

        acc_sum = tl.zeros([], dtype=tl.float32)
        acc_sum_sq = tl.zeros([], dtype=tl.float32)

        row_base = x_ptr + r * stride_row
        row_base = tl.multiple_of(row_base, BLOCK_SIZE)

        # Fully unrolled column-tile loop – N_COLS % BLOCK_SIZE == 0
        for col_start in tl.static_range(0, N_COLS, BLOCK_SIZE):
            offsets = col_start + tl.arange(0, BLOCK_SIZE)
            vals = tl.load(row_base + offsets, cache_modifier='.cg')
            acc_sum += tl.sum(vals, axis=0)
            acc_sum_sq += tl.sum(vals * vals, axis=0)

        mean = acc_sum / N_f
        var = acc_sum_sq / N_f - mean * mean

        # Store into the two-row output tensor
        tl.store(out_ptr + r, mean)
        tl.store(out_ptr + N_ROWS + r, var)