import torch
import triton
import triton.language as tl
import math


@triton.jit
def kl_rowsum_kernel(
    log_p_ptr, q_ptr, row_sums_ptr,
    batch_size, cols, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)   # row index
    row_start = pid * cols

    sum_row = 0.0
    for start in range(0, cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < cols

        lp = tl.load(log_p_ptr + row_start + offsets, mask=mask, other=0.0)
        q = tl.load(q_ptr + row_start + offsets, mask=mask, other=0.0)

        # Avoid log(0) by setting log_q = 0 when q == 0
        log_q = tl.where(q > 0, tl.log(q), 0.0)
        term = tl.where(q > 0, q * (log_q - lp), 0.0)

        # Sum the term under the mask (unmasked entries contribute 0)
        sum_row += tl.sum(tl.where(mask, term, 0.0))

    tl.store(row_sums_ptr + pid, sum_row)


@triton.jit
def row_sums_reduce_kernel(
    row_sums_ptr, output_ptr,
    n, batch_size, BLOCK_SIZE: tl.constexpr
):
    # Only one program, BLOCK_SIZE threads.
    acc = 0.0
    for start in range(0, n, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        vals = tl.load(row_sums_ptr + offsets, mask=mask, other=0.0)
        acc += tl.sum(vals)

    # Write result (first thread)
    if tl.arange(0, BLOCK_SIZE)[0] == 0:
        tl.store(output_ptr, acc / batch_size)


def triton_run(log_p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    device = log_p.device
    batch_size, cols = log_p.shape

    # Allocate output (scalar)
    output = torch.empty((), dtype=torch.float32, device=device)

    # First pass: reduce each row to a scalar
    row_sums = torch.empty(batch_size, dtype=torch.float32, device=device)

    BLOCK_ROW = 2048
    grid = (batch_size,)
    kl_rowsum_kernel[grid](
        log_p, q, row_sums,
        batch_size, cols, BLOCK_SIZE=BLOCK_ROW,
        num_warps=8, num_stages=2
    )

    # Second pass: sum all row sums and divide by batch_size
    BLOCK_REDUCE = 1024
    grid2 = (1,)
    row_sums_reduce_kernel[grid2](
        row_sums, output,
        batch_size, batch_size, BLOCK_SIZE=BLOCK_REDUCE,
        num_warps=4, num_stages=2
    )

    return output