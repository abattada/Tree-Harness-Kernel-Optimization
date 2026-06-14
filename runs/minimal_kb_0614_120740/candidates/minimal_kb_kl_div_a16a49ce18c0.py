import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 1: per-row KL divergence sum over columns
#   term = q * (log(q) - log_p), sum over columns, store row sum.
# ---------------------------------------------------------------------------
@triton.jit
def row_kl_kernel(
    log_p_ptr,          # [rows, cols]
    q_ptr,              # [rows, cols]
    row_sum_ptr,        # [rows]
    rows: tl.constexpr,
    cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * cols

    acc = tl.zeros([], dtype=tl.float32)

    # Loop over columns in blocks (only one iteration if BLOCK_SIZE == cols)
    for col_start in range(0, cols, BLOCK_SIZE):
        offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < cols

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

        # Safe computation: if q == 0, term is 0 (avoid 0 * (-inf))
        term = tl.where(q_vals > 0, q_vals * (tl.log(q_vals) - log_p_vals), 0.0)
        acc += tl.sum(term)

    tl.store(row_sum_ptr + pid, acc)


# ---------------------------------------------------------------------------
# Kernel 2: reduce row sums to a single scalar, divide by number of rows
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
        vals = tl.load(
            row_sum_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy='evict_first',
        )
        total += tl.sum(vals)

    # All threads hold the same total; store the mean (= total / rows)
    tl.store(scalar_ptr, total / rows)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def triton_run(log_p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence with reduction='batchmean'.
    Input shapes: (8192, 8192) for both log_p and q.
    Returns a scalar tensor (0-D).
    """
    assert log_p.shape == q.shape
    rows, cols = log_p.shape

    # Allocate intermediate row sums and final scalar output
    row_sum = torch.empty(rows, dtype=torch.float32, device=log_p.device)
    scalar_out = torch.empty(1, dtype=torch.float32, device=log_p.device)

    # Launch parameters for row kernel
    # BLOCK_SIZE = cols (8192) so each program processes one full row in one iteration
    BLOCK_SIZE_ROW = 8192
    grid_row = (rows,)
    row_kl_kernel[grid_row](
        log_p, q, row_sum,
        rows, cols, BLOCK_SIZE_ROW,
        num_warps=4,
    )

    # Launch parameters for reduction kernel
    BLOCK_SIZE_REDUCE = 2048
    grid_reduce = (1,)  # single program
    reduce_row_sums_kernel[grid_reduce](
        row_sum, scalar_out,
        rows, BLOCK_SIZE_REDUCE,
        num_warps=1,
    )

    # Return scalar tensor (squeeze the [1] dimension)
    return scalar_out[0]