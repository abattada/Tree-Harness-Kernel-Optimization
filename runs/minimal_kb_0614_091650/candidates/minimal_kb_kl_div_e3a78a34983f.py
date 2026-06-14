import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 1: per-row KL divergence (sum over columns of q*(log(q)-log_p))
# ---------------------------------------------------------------------------
@triton.jit
def row_kl_kernel(
    log_p_ptr,          # [rows, cols]
    q_ptr,              # [rows, cols]
    row_sum_ptr,        # [rows]  output: per-row sums
    rows: tl.constexpr,
    cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)               # one program per row
    row_start = pid * cols

    acc = tl.zeros([], dtype=tl.float32)

    # Loop over columns in blocks
    for col_start in range(0, cols, BLOCK_SIZE):
        offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < cols

        q_vals = tl.load(q_ptr + row_start + offsets, mask=mask, other=0.0)
        log_p_vals = tl.load(log_p_ptr + row_start + offsets, mask=mask, other=0.0)

        # term = q * (log(q) - log_p), with safe handling for q == 0
        term = tl.where(q_vals > 0, q_vals * (tl.log(q_vals) - log_p_vals), 0.0)
        acc += tl.sum(term)

    tl.store(row_sum_ptr + pid, acc)


# ---------------------------------------------------------------------------
# Kernel 2: reduction of row sums to a single scalar, then divide by rows
# ---------------------------------------------------------------------------
@triton.jit
def reduce_row_sums_kernel(
    row_sum_ptr,        # [rows]
    scalar_ptr,         # [1]
    rows: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    total = tl.zeros([], dtype=tl.float32)
    for start in range(0, rows, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < rows
        vals = tl.load(row_sum_ptr + offsets, mask=mask, other=0.0)
        total += tl.sum(vals)

    # All threads hold the same total; store it (simply all threads write the same)
    tl.store(scalar_ptr, total / rows)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def triton_run(log_p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Compute Kullback–Leibler divergence with reduction='batchmean'.
    Input shapes are both (8192, 8192).
    Returns a scalar tensor.
    """
    assert log_p.shape == q.shape
    rows, cols = log_p.shape

    # Temporary buffer for per-row sums
    row_sum = torch.empty(rows, dtype=torch.float32, device=log_p.device)
    # Output scalar (size 1 then squeezed)
    scalar_out = torch.empty(1, dtype=torch.float32, device=log_p.device)

    # Tuning knobs (can be adjusted later)
    BLOCK_SIZE = 1024
    NUM_WARPS = 8

    # Kernel 1: one program per row
    grid_row = (rows,)
    row_kl_kernel[grid_row](
        log_p, q, row_sum,
        rows, cols, BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )

    # Kernel 2: single program to reduce row sums
    grid_red = (1,)
    reduce_row_sums_kernel[grid_red](
        row_sum, scalar_out,
        rows, BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )

    return scalar_out.squeeze(0)