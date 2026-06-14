import torch
import triton
import triton.language as tl


@triton.jit
def kl_multirow_kernel(
    log_p_ptr,             # [rows, cols]
    q_ptr,                 # [rows, cols]
    row_sum_ptr,           # [grid_rows]  partial sums
    rows: tl.constexpr,
    cols: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Each program processes ROWS_PER_PROG consecutive rows.
    It accumulates the KL divergence sum over those rows,
    storing a single scalar per program for later reduction.
    """
    pid = tl.program_id(0)
    start_row = pid * ROWS_PER_PROG
    prog_sum = tl.zeros([], dtype=tl.float32)

    for r in range(ROWS_PER_PROG):
        row = start_row + r
        if row >= rows:
            break
        row_start = row * cols
        acc_row = tl.zeros([], dtype=tl.float32)

        # Inner loop over columns (same as original per-row kernel)
        for col_start in range(0, cols, BLOCK_SIZE):
            offsets = tl.max_contiguous(
                tl.multiple_of(col_start + tl.arange(0, BLOCK_SIZE), BLOCK_SIZE),
                BLOCK_SIZE,
            )
            mask = offsets < cols

            q_vals = tl.load(
                q_ptr + row_start + offsets,
                mask=mask,
                other=0.0,
                eviction_policy='evict_first',
            )
            log_p_vals = tl.load(
                log_p_ptr + row_start + offsets,
                mask=mask,
                other=0.0,
                eviction_policy='evict_first',
            )
            term = tl.where(q_vals > 0.0, q_vals * (tl.log(q_vals) - log_p_vals), 0.0)
            acc_row += tl.sum(term)

        prog_sum += acc_row

    tl.store(row_sum_ptr + pid, prog_sum)


@triton.jit
def reduce_kernel(
    partial_sum_ptr,       # [elems]
    scalar_ptr,            # [1]
    elems: tl.constexpr,
    total_rows: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Reduce an array of partial sums to a single scalar,
    then divide by total_rows to compute the batchmean.
    """
    total = tl.zeros([], dtype=tl.float32)
    for start in range(0, elems, BLOCK_SIZE):
        offsets = tl.max_contiguous(
            tl.multiple_of(start + tl.arange(0, BLOCK_SIZE), BLOCK_SIZE),
            BLOCK_SIZE,
        )
        mask = offsets < elems
        vals = tl.load(
            partial_sum_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy='evict_first',
        )
        total += tl.sum(vals)

    tl.store(scalar_ptr, total / total_rows)


def triton_run(log_p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence with reduction='batchmean'.
    Inputs: log_p, q  — both f32 [8192, 8192]
    Returns: f32 scalar tensor
    """
    assert log_p.shape == q.shape, "Input shapes must match"
    assert log_p.dtype == q.dtype == torch.float32, "Inputs must be float32"
    rows, cols = log_p.shape

    ROWS_PER_PROG = 16
    grid_rows = (rows + ROWS_PER_PROG - 1) // ROWS_PER_PROG

    # Intermediate array for per-program partial sums (size grid_rows)
    partial_sums = torch.empty(grid_rows, dtype=torch.float32, device=log_p.device)
    scalar_out = torch.empty(1, dtype=torch.float32, device=log_p.device)

    BLOCK_SIZE_ROW = 1024
    kl_multirow_kernel[(grid_rows,)](
        log_p, q, partial_sums,
        rows=rows, cols=cols,
        ROWS_PER_PROG=ROWS_PER_PROG,
        BLOCK_SIZE=BLOCK_SIZE_ROW,
        num_warps=8,
        num_stages=2,
    )

    BLOCK_SIZE_REDUCE = 512   # grid_rows = 512 → single iteration
    reduce_kernel[(1,)](
        partial_sums, scalar_out,
        elems=grid_rows,
        total_rows=rows,
        BLOCK_SIZE=BLOCK_SIZE_REDUCE,
        num_warps=4,
        num_stages=2,
    )

    return scalar_out.squeeze()