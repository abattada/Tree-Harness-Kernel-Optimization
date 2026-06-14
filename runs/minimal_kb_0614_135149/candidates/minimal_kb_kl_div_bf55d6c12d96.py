import torch
import triton
import triton.language as tl


@triton.jit
def row_kl_kernel(
    log_p_ptr,          # f32 [rows, cols]
    q_ptr,              # f32 [rows, cols]
    row_sum_ptr,        # f32 [rows]
    rows: tl.constexpr,
    cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute per-row sum of q * (log(q) - log_p) with safe handling of q==0."""
    pid = tl.program_id(0)
    row_start = pid * cols

    acc = tl.zeros([], dtype=tl.float32)
    for col_start in range(0, cols, BLOCK_SIZE):
        offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < cols

        q_vals = tl.load(
            q_ptr + row_start + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        )
        log_p_vals = tl.load(
            log_p_ptr + row_start + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        )

        term = tl.where(q_vals > 0, q_vals * (tl.log(q_vals) - log_p_vals), 0.0)
        acc += tl.sum(term)

    tl.store(row_sum_ptr + pid, acc)


@triton.jit
def reduce_scalar_kernel(
    row_sum_ptr,        # f32 [rows]
    scalar_ptr,         # f32 [1]
    rows: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Reduce row-wise sums to a single scalar, then divide by rows (batchmean)."""
    total = tl.zeros([], dtype=tl.float32)
    for start in range(0, rows, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < rows
        vals = tl.load(
            row_sum_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        )
        total += tl.sum(vals)

    tl.store(scalar_ptr, total / rows)


def triton_run(log_p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Compute batchmean KL divergence: sum(q*(log(q)-log_p)) / rows.
    Inputs: log_p (f32[8192, 8192]), q (f32[8192, 8192])
    Returns: scalar f32[] tensor.
    """
    rows, cols = log_p.shape
    assert q.shape == (rows, cols)

    # Intermediate row sums
    row_sum = torch.empty(rows, dtype=torch.float32, device=log_p.device)
    # Final scalar
    scalar_out = torch.empty(1, dtype=torch.float32, device=log_p.device)

    # Row KL kernel – process one row per program, with a tunable column block size
    BLOCK_ROW = 1024
    grid_row = (rows,)
    row_kl_kernel[grid_row](
        log_p, q, row_sum,
        rows, cols,
        BLOCK_SIZE=BLOCK_ROW,
        num_warps=32,
        num_stages=2,
    )

    # Reduction kernel – single block sums rows and computes batchmean
    BLOCK_RED = 512
    reduce_scalar_kernel[(1,)](
        row_sum, scalar_out,
        rows,
        BLOCK_SIZE=BLOCK_RED,
        num_warps=16,
        num_stages=1,
    )

    return scalar_out