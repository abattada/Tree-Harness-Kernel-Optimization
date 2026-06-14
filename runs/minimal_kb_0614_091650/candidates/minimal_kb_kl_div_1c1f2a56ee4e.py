import torch
import triton
import triton.language as tl

@triton.jit
def kl_div_row_kernel(
    log_p_ptr,          # [rows, cols]
    q_ptr,              # [rows, cols]
    row_sum_ptr,        # [rows]
    cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)   # each program handles one row
    row_start = pid * cols
    acc = tl.zeros([], dtype=tl.float32)
    for col_start in range(0, cols, BLOCK_SIZE):
        offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < cols
        q_vals = tl.load(q_ptr + row_start + offsets, mask, other=0.0)
        log_p_vals = tl.load(log_p_ptr + row_start + offsets, mask, other=0.0)
        # q * (log(q) - log_p); log(0) is -inf, but we zero the product
        term = tl.where(q_vals > 0, q_vals * (tl.log(q_vals) - log_p_vals), 0.0)
        acc += tl.sum(term)
    tl.store(row_sum_ptr + pid, acc)


@triton.jit
def reduce_kernel(
    row_sum_ptr,        # [rows]
    scalar_ptr,         # [1]
    rows: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    total = tl.zeros([], dtype=tl.float32)
    for start in range(0, rows, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < rows
        vals = tl.load(row_sum_ptr + offsets, mask, other=0.0)
        total += tl.sum(vals)
    # All threads have the same total; store by thread 0 (all write same value)
    tl.store(scalar_ptr, total / rows)


def triton_run(log_p, q) -> torch.Tensor:
    rows, cols = log_p.shape
    assert q.shape == (rows, cols)

    row_sum = torch.empty(rows, dtype=torch.float32, device=log_p.device)
    scalar_out = torch.empty(1, dtype=torch.float32, device=log_p.device)

    BLOCK_SIZE = 1024
    # First kernel: one program per row, each processes cols in BLOCK_SIZE chunks
    grid_row = (rows,)
    kl_div_row_kernel[grid_row](
        log_p, q, row_sum,
        cols, BLOCK_SIZE,
        num_warps=8,
    )
    # Second kernel: reduce row sums to scalar
    grid_red = (1,)
    reduce_kernel[grid_red](
        row_sum, scalar_out,
        rows, BLOCK_SIZE,
        num_warps=8,
    )
    return scalar_out.squeeze(0)