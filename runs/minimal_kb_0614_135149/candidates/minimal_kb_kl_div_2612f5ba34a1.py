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
        sum_{j} q_j * (log q_j - log_p_j)
    avoiding NaN from 0 * log(0).
    Assumes cols is a multiple of BLOCK_SIZE, so no boundary masking needed.
    """
    pid = tl.program_id(0)
    row_start = pid * cols
    acc = tl.zeros([], dtype=tl.float32)

    for col_start in range(0, cols, BLOCK_SIZE):
        offsets = tl.max_contiguous(
            tl.multiple_of(col_start + tl.arange(0, BLOCK_SIZE), BLOCK_SIZE),
            BLOCK_SIZE,
        )
        # No mask: cols % BLOCK_SIZE == 0
        q_vals = tl.load(
            q_ptr + row_start + offsets,
            eviction_policy='evict_first',
        )
        log_p_vals = tl.load(
            log_p_ptr + row_start + offsets,
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
    Assumes rows is a multiple of BLOCK_SIZE, no masking needed.
    """
    total = tl.zeros([], dtype=tl.float32)
    for start in range(0, rows, BLOCK_SIZE):
        offsets = tl.max_contiguous(
            tl.multiple_of(start + tl.arange(0, BLOCK_SIZE), BLOCK_SIZE),
            BLOCK_SIZE,
        )
        vals = tl.load(row_sum_ptr + offsets)
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

    # Block sizes chosen as divisors of the fixed 8192 x 8192 shape.
    BLOCK_SIZE_ROW = 2048
    BLOCK_SIZE_RED = 4096
    assert cols % BLOCK_SIZE_ROW == 0, "cols must be divisible by BLOCK_SIZE_ROW"
    assert rows % BLOCK_SIZE_RED == 0, "rows must be divisible by BLOCK_SIZE_RED"

    # Intermediate per-row sums and final scalar
    row_sum = torch.empty(rows, dtype=torch.float32, device=log_p.device)
    scalar_out = torch.empty(1, dtype=torch.float32, device=log_p.device)

    # Kernel 1: per-row KL divergence sum
    kl_row_kernel[(rows,)](
        log_p, q, row_sum,
        cols=cols, BLOCK_SIZE=BLOCK_SIZE_ROW,
        num_warps=16, num_stages=2,
    )

    # Kernel 2: reduce row sums to scalar batchmean
    reduce_kernel[(1,)](
        row_sum, scalar_out,
        rows=rows, BLOCK_SIZE=BLOCK_SIZE_RED,
        num_warps=4, num_stages=2,
    )

    return scalar_out.squeeze()