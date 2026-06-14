import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 1: per-block KL divergence sum over multiple rows.
#   log_p:  [rows, cols] f32
#   q:      [rows, cols] f32
#   block_sum: [num_blocks] f32, stores partial sums
# ---------------------------------------------------------------------------
@triton.jit
def row_kl_multi_kernel(
    log_p_ptr,
    q_ptr,
    block_sum_ptr,
    rows: tl.constexpr,
    cols: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_row_start = pid * ROWS_PER_PROG

    # Full-row offset vector – no mask needed when cols == BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)

    acc = tl.zeros([], dtype=tl.float32)
    for r in range(ROWS_PER_PROG):
        row_idx = block_row_start + r
        row_start = row_idx * cols

        q_vals = tl.load(
            q_ptr + row_start + offsets,
            eviction_policy='evict_first',
        )
        log_p_vals = tl.load(
            log_p_ptr + row_start + offsets,
            eviction_policy='evict_first',
        )

        # term = q * (log(q) - log_p) ; safely handle q == 0
        term = tl.where(q_vals > 0.0,
                        q_vals * (tl.log(q_vals) - log_p_vals),
                        0.0)
        acc += tl.sum(term)

    tl.store(block_sum_ptr + pid, acc)


# ---------------------------------------------------------------------------
# Kernel 2: reduce block sums to a scalar (batchmean).
# ---------------------------------------------------------------------------
@triton.jit
def reduce_scalar_kernel(
    block_sum_ptr,
    scalar_ptr,
    num_blocks: tl.constexpr,
    rows: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    # num_blocks == BLOCK_SIZE, so no mask needed
    vals = tl.load(
        block_sum_ptr + offsets,
        eviction_policy='evict_first',
    )
    total = tl.sum(vals)
    tl.store(scalar_ptr, total / rows)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def triton_run(log_p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    KL divergence with batchmean reduction:
        sum(q * (log q - log_p)) / batch_size.
    Inputs: both [8192, 8192], float32.
    Returns: scalar float32 tensor.
    """
    rows, cols = log_p.shape
    assert log_p.shape == q.shape, "Input shapes must match"
    assert log_p.dtype == torch.float32, "Inputs must be float32"

    ROWS_PER_PROG = 8
    assert rows % ROWS_PER_PROG == 0, "row count must be divisible by ROWS_PER_PROG"
    num_blocks = rows // ROWS_PER_PROG

    block_sum = torch.empty(num_blocks, dtype=torch.float32, device=log_p.device)
    scalar_out = torch.empty(1, dtype=torch.float32, device=log_p.device)

    BLOCK_COLS = cols        # 8192 – process one full row per iteration
    BLOCK_REDUCE = num_blocks  # 1024 – matches the reduction array size

    # Kernel 1: compute partial sums, ROWS_PER_PROG rows per block
    row_kl_multi_kernel[(num_blocks,)](
        log_p, q, block_sum,
        rows=rows, cols=cols, ROWS_PER_PROG=ROWS_PER_PROG, BLOCK_SIZE=BLOCK_COLS,
        num_warps=16,
    )

    # Kernel 2: reduce partial sums to scalar and divide by rows
    reduce_scalar_kernel[(1,)](
        block_sum, scalar_out,
        num_blocks=num_blocks, rows=rows, BLOCK_SIZE=BLOCK_REDUCE,
        num_warps=2,
    )

    return scalar_out.squeeze()