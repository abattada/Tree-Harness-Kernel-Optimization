import torch
import triton
import triton.language as tl

@triton.jit
def kl_div_row_kernel(
    log_p_ptr,          # [rows, cols]
    q_ptr,              # [rows, cols]
    row_sum_ptr,        # [rows]
    cols: tl.constexpr,    # number of columns
    BLOCK_SIZE: tl.constexpr,  # must equal cols (no loop, no mask)
):
    pid = tl.program_id(0)          # each program handles one row
    row_start = pid * cols
    offsets = tl.arange(0, BLOCK_SIZE)   # contiguous, full row
    # Load entire row – no mask needed because BLOCK_SIZE == cols
    q_vals = tl.load(q_ptr + row_start + offsets)
    log_p_vals = tl.load(log_p_ptr + row_start + offsets)
    # Compute term: q * (log(q) - log_p)
    # For q == 0, contribution is 0 by convention.
    log_q = tl.where(q_vals > 0, tl.log(q_vals), 0.0)
    term = q_vals * (log_q - log_p_vals)
    acc = tl.sum(term)
    tl.store(row_sum_ptr + pid, acc)


@triton.jit
def reduce_kernel(
    row_sum_ptr,        # [rows]
    scalar_ptr,         # [1]
    rows: tl.constexpr,    # number of rows
    BLOCK_SIZE: tl.constexpr,  # must equal rows (no loop, no mask)
):
    offsets = tl.arange(0, BLOCK_SIZE)   # full array
    vals = tl.load(row_sum_ptr + offsets)
    total = tl.sum(vals)
    # All threads have the same total; store by one (all write same value)
    tl.store(scalar_ptr, total / rows)


def triton_run(log_p, q) -> torch.Tensor:
    rows, cols = log_p.shape
    assert q.shape == (rows, cols)

    row_sum = torch.empty(rows, dtype=torch.float32, device=log_p.device)
    scalar_out = torch.empty(1, dtype=torch.float32, device=log_p.device)

    # Row kernel: BLOCK_SIZE = cols (drop loop and mask)
    grid_row = (rows,)
    kl_div_row_kernel[grid_row](
        log_p, q, row_sum,
        cols, cols,           # cols as shape, BLOCK_SIZE = cols
        num_warps=8,
    )
    # Reduction kernel: BLOCK_SIZE = rows (drop loop and mask)
    grid_red = (1,)
    reduce_kernel[grid_red](
        row_sum, scalar_out,
        rows, rows,           # rows as shape, BLOCK_SIZE = rows
        num_warps=8,
    )
    return scalar_out.squeeze(0)