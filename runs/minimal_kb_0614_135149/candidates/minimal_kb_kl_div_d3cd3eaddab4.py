import torch
import triton
import triton.language as tl


@triton.jit
def kl_row_kernel(
    log_p_ptr,
    q_ptr,
    row_sum_ptr,
    cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Each program computes the KL divergence sum over one row:
        sum_j  q_j * (log q_j - log_p_j)
    avoiding NaN from 0 * log(0).
    """
    pid = tl.program_id(0)
    row_start = pid * cols
    acc = tl.zeros([], dtype=tl.float32)

    for col_start in range(0, cols, BLOCK_SIZE):
        offsets = tl.max_contiguous(
            tl.multiple_of(col_start + tl.arange(0, BLOCK_SIZE), BLOCK_SIZE),
            BLOCK_SIZE,
        )
        mask = offsets < cols

        # Streaming access: each row is read once, evict_first is appropriate.
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
        acc += tl.sum(term)

    tl.store(row_sum_ptr + pid, acc)


@triton.jit
def reduce_kernel(
    row_sum_ptr,
    scalar_ptr,
    rows: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Single-block reduction: sum all row sums and divide by rows to get batchmean.
    """
    total = tl.zeros([], dtype=tl.float32)
    for start in range(0, rows, BLOCK_SIZE):
        offsets = tl.max_contiguous(
            tl.multiple_of(start + tl.arange(0, BLOCK_SIZE), BLOCK_SIZE),
            BLOCK_SIZE,
        )
        mask = offsets < rows
        vals = tl.load(row_sum_ptr + offsets, mask=mask, other=0.0)
        total += tl.sum(vals)

    tl.store(scalar_ptr, total / rows)


def triton_run(log_p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence with reduction='batchmean'.
    Inputs: log_p, q  — both f32[8192, 8192]
    Returns: f32[] scalar
    """
    assert log_p.shape == q.shape, "Input shapes must match"
    assert log_p.dtype == q.dtype == torch.float32, "Inputs must be float32"
    rows, cols = log_p.shape

    # Intermediate per-row sums
    row_sum = torch.empty(rows, dtype=torch.float32, device=log_p.device)
    scalar_out = torch.empty(1, dtype=torch.float32, device=log_p.device)

    # Kernel 1: per-row KL divergence sum
    # Increased block size to 2048 halves the number of loop iterations
    # while keeping occupancy high.
    BLOCK_SIZE_ROW: tl.constexpr = 2048
    kl_row_kernel[(rows,)](
        log_p, q, row_sum,
        cols=cols, BLOCK_SIZE=BLOCK_SIZE_ROW,
        num_warps=8, num_stages=2,
    )

    # Kernel 2: reduce row sums to scalar batchmean
    # A block size of 1024 with 4 warps is still fine; the loop overhead is tiny.
    BLOCK_SIZE_REDUCE: tl.constexpr = 1024
    reduce_kernel[(1,)](
        row_sum, scalar_out,
        rows=rows, BLOCK_SIZE=BLOCK_SIZE_REDUCE,
        num_warps=4, num_stages=2,
    )

    return scalar_out.squeeze()