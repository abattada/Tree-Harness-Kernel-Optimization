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
    Each program processes one row: sum of q * (log(q) - log_p) over columns.
    cols is guaranteed to be a multiple of BLOCK_SIZE, so no boundary mask is
    needed.  Pipeline depth increased to 4 stages to hide load latency.
    """
    pid = tl.program_id(0)
    row_start = pid * cols
    acc = tl.zeros([], dtype=tl.float32)

    for col_start in range(0, cols, BLOCK_SIZE):
        offsets = col_start + tl.arange(0, BLOCK_SIZE)
        # All offsets are strictly < cols → mask is unnecessary.
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
    Single‑block reduction: sum all row‑wise partials and divide by rows.
    rows is a multiple of BLOCK_SIZE → no mask required.
    """
    total = tl.zeros([], dtype=tl.float32)
    for start in range(0, rows, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        vals = tl.load(row_sum_ptr + offsets)
        total += tl.sum(vals)

    tl.store(scalar_ptr, total / rows)


def triton_run(log_p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Compute Kullback–Leibler divergence with reduction='batchmean'.
    Inputs: log_p, q — both f32[8192, 8192]
    Returns: f32[] scalar
    """
    assert log_p.shape == q.shape, "Input shapes must match"
    assert log_p.dtype == q.dtype == torch.float32, "Inputs must be float32"
    rows, cols = log_p.shape

    row_sum = torch.empty(rows, dtype=torch.float32, device=log_p.device)
    scalar_out = torch.empty(1, dtype=torch.float32, device=log_p.device)

    # Per‑row KL sum. cols = 8192 = 8 × 1024 → no remainder.
    BLOCK_SIZE_ROW: tl.constexpr = 1024
    kl_row_kernel[(rows,)](
        log_p, q, row_sum,
        cols=cols, BLOCK_SIZE=BLOCK_SIZE_ROW,
        num_warps=8, num_stages=4,       # deeper pipeline for better latency hiding
    )

    # Reduce row sums to scalar. rows = 8192 = 8 × 1024 → no remainder.
    BLOCK_SIZE_REDUCE: tl.constexpr = 1024
    reduce_kernel[(1,)](
        row_sum, scalar_out,
        rows=rows, BLOCK_SIZE=BLOCK_SIZE_REDUCE,
        num_warps=1, num_stages=2,       # minimal config for the tiny reduction
    )

    return scalar_out.squeeze()