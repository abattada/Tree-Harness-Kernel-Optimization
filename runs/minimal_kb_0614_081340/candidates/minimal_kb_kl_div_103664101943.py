import torch
import triton
import triton.language as tl
import math

# ---------------------------------------------------------------------------
# Kernel 1: compute per-row partial sum of q * (log(q) - log_p)
# ---------------------------------------------------------------------------
@triton.jit
def kl_div_row_sum_kernel(
    log_p_ptr, q_ptr, partial_ptr,
    n_rows, n_cols,
    stride_log_p, stride_q,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid
    if row >= n_rows:
        return

    # base pointers for this row
    log_p_row = log_p_ptr + row * stride_log_p
    q_row = q_ptr + row * stride_q

    # accumulate in a register
    acc = tl.zeros([1], dtype=tl.float32)

    # iterate over columns in tiles
    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        # load log_p and q
        lp = tl.load(log_p_row + col_offsets, mask=mask, other=0.0)
        qv = tl.load(q_row + col_offsets, mask=mask, other=0.0)

        # compute log(q) safely (zero for q == 0)
        log_q = tl.where(qv > 0, tl.log(qv), 0.0)

        # elementwise contribution: q * (log_q - log_p)
        contrib = qv * (log_q - lp)
        acc += tl.sum(contrib, axis=0)

    # write partial sum
    tl.store(partial_ptr + row, acc)

# ---------------------------------------------------------------------------
# Kernel 2: sum all partial sums into a scalar
# ---------------------------------------------------------------------------
@triton.jit
def reduce_sum_kernel(
    partial_ptr, output_ptr,
    n_partials,
    BLOCK_SIZE: tl.constexpr,
):
    # single program reduces the whole partial array
    pid = tl.program_id(0)
    if pid != 0:
        return

    # shared memory for tree reduction
    smem = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # load partials into shared memory
    for i in range(0, n_partials, BLOCK_SIZE):
        offs = tl.arange(0, BLOCK_SIZE)
        mask = (i + offs) < n_partials
        val = tl.load(partial_ptr + i + offs, mask=mask, other=0.0)
        smem[:] = smem + val

    # parallel reduction in shared memory
    for stride in range(BLOCK_SIZE // 2, 0, stride // 2):
        # ensure we stay within the number of active lanes
        if tl.program_id(0) == 0:
            # We'll use a loop over iterations, but simpler: use tl.reduce
            pass
    # Actually, tree reduction in shared memory:
    # We have BLOCK_SIZE threads; each thread holds one element.
    # Use a loop that halves the stride each time.
    # However, we only have one block (grid=(1,)), so we can just use a simple loop.
    # Since n_partials may be less than BLOCK_SIZE, we need to handle it.
    # Let's instead use a for loop based on powers of two.
    size = n_partials
    while size > 1:
        half = size // 2
        for i in range(half):
            smem[i] = smem[i] + smem[i + half]
        size = half
        # synchronize (implicit in Triton for same block, but use tl.debug_barrier() for safety)
        tl.debug_barrier()
    # now smem[0] has total sum
    total = smem[0]
    # write scalar
    tl.store(output_ptr, total)

@triton.jit
def reduce_sum_kernel_v2(
    partial_ptr, output_ptr,
    n_partials,
    BLOCK_SIZE: tl.constexpr,
):
    # Simpler: use a single block with tl.reduce
    pid = tl.program_id(0)
    if pid != 0:
        return
    sum_acc = tl.zeros([1], dtype=tl.float32)
    for i in range(0, n_partials, BLOCK_SIZE):
        offs = tl.arange(0, BLOCK_SIZE)
        mask = (i + offs) < n_partials
        val = tl.load(partial_ptr + i + offs, mask=mask, other=0.0)
        sum_acc += tl.sum(val)
    tl.store(output_ptr, sum_acc)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def triton_run(log_p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    assert log_p.shape == q.shape and log_p.dtype == torch.float32 and q.dtype == torch.float32
    n_rows, n_cols = log_p.shape
    # output scalar
    output = torch.empty(1, dtype=torch.float32, device=log_p.device)
    # partial sums per row
    partial = torch.empty(n_rows, dtype=torch.float32, device=log_p.device)

    # Launch first kernel
    BLOCK_SIZE = 1024  # can be tuned
    grid_rows = (n_rows,)
    kl_div_row_sum_kernel[grid_rows](
        log_p, q, partial,
        n_rows, n_cols,
        log_p.stride(0), q.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Launch second kernel: sum partials and divide by batch size
    # Use a single block, BLOCK_SIZE = 1024
    BLOCK_REDUCE = 1024
    grid_reduce = (1,)
    reduce_sum_kernel_v2[grid_reduce](
        partial, output,
        n_rows,
        BLOCK_SIZE=BLOCK_REDUCE,
    )

    # Divide by batch size (batchmean)
    output[0] = output[0] / n_rows
    return output