import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 1: per‑row KL divergence with grid‑stride loop over rows.
# Each program processes ROWS_PER_PROG rows, reducing launch overhead.
# ---------------------------------------------------------------------------
@triton.jit
def row_kl_kernel(
    log_p_ptr,               # f32 [rows, cols]
    q_ptr,                   # f32 [rows, cols]
    row_sum_ptr,             # f32 [rows]  (output)
    rows: tl.constexpr,
    cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start_group = pid * ROWS_PER_PROG

    for r in range(ROWS_PER_PROG):
        row_idx = row_start_group + r
        if row_idx >= rows:
            continue

        row_start = row_idx * cols
        acc = tl.zeros([], dtype=tl.float32)

        # Column loop: BLOCK_SIZE divides cols exactly, so no mask needed
        for col_start in range(0, cols, BLOCK_SIZE):
            offsets = col_start + tl.arange(0, BLOCK_SIZE)
            q_vals = tl.load(
                q_ptr + row_start + offsets,
                eviction_policy='evict_first',
            )
            log_p_vals = tl.load(
                log_p_ptr + row_start + offsets,
                eviction_policy='evict_first',
            )

            # term = q * (log q - log_p), safely skip q == 0 (avoids NaN)
            term = tl.where(q_vals > 0.0,
                            q_vals * (tl.log(q_vals) - log_p_vals), 0.0)
            acc += tl.sum(term)

        tl.store(row_sum_ptr + row_idx, acc)


# ---------------------------------------------------------------------------
# Kernel 2: reduce per‑row sums to a scalar (batchmean = sum / rows)
# ---------------------------------------------------------------------------
@triton.jit
def reduce_scalar_kernel(
    row_sum_ptr,              # f32 [rows]
    scalar_ptr,               # f32 [1]
    rows: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    total = tl.zeros([], dtype=tl.float32)
    for start in range(0, rows, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < rows
        vals = tl.load(
            row_sum_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy='evict_first',
        )
        total += tl.sum(vals)

    tl.store(scalar_ptr, total / rows)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def triton_run(log_p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    KL divergence with batchmean reduction: sum(q*(log q - log_p)) / batch_size.
    Inputs: both [8192, 8192] float32.
    Returns: 0‑D scalar float32 tensor.
    """
    assert log_p.shape == q.shape, "Input shapes must match"
    assert log_p.dtype == torch.float32, "Inputs must be float32"
    rows, cols = log_p.shape

    # Allocations
    row_sum = torch.empty(rows, dtype=torch.float32, device=log_p.device)
    scalar_out = torch.empty(1, dtype=torch.float32, device=log_p.device)

    # Tuning constants for 8192×8192
    BLOCK_SIZE = 2048         # divides 8192, good balance of occupancy vs. registers
    ROWS_PER_PROG = 8         # reduce launch count, amortize overhead
    REDUCE_BLOCK = 4096       # two passes over 8192 rows, single‑warp overhead is tiny

    # Row kernel: coarse grid, each program handles multiple rows
    grid = (triton.cdiv(rows, ROWS_PER_PROG),)
    row_kl_kernel[grid](
        log_p, q, row_sum,
        rows=rows, cols=cols,
        BLOCK_SIZE=BLOCK_SIZE,
        ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=16, num_stages=2,
    )

    # Reduction kernel: single program over the row‑sum array
    reduce_scalar_kernel[(1,)](
        row_sum, scalar_out,
        rows=rows,
        BLOCK_SIZE=REDUCE_BLOCK,
        num_warps=1,
    )

    return scalar_out.squeeze()