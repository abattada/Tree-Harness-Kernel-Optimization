import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 1: per-row KL divergence, each program processes a block of rows
# to reduce grid size and improve occupancy.
# ---------------------------------------------------------------------------
@triton.jit
def row_kl_kernel(
    log_p_ptr,          # [rows, cols]
    q_ptr,              # [rows, cols]
    row_sum_ptr,        # [rows]
    rows: tl.constexpr,
    cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)

    # Each program processes ROWS_PER_PROG consecutive rows
    row_start = pid * ROWS_PER_PROG
    row_end = tl.minimum(row_start + ROWS_PER_PROG, rows)

    # Iterate over rows assigned to this program
    for r in range(row_start, row_end):
        acc = tl.zeros([], dtype=tl.float32)

        # Single iteration since BLOCK_SIZE == cols
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < cols

        # Load q and log_p with streaming cache hints
        q_vals = tl.load(
            q_ptr + r * cols + offsets,
            mask=mask,
            other=0.0,
            eviction_policy='evict_first',
        )
        log_p_vals = tl.load(
            log_p_ptr + r * cols + offsets,
            mask=mask,
            other=0.0,
            eviction_policy='evict_first',
        )

        # term = q * (log(q) - log_p), avoid 0 * (-inf)
        term = tl.where(q_vals > 0, q_vals * (tl.log(q_vals) - log_p_vals), 0.0)
        acc += tl.sum(term)

        # Write row sum
        tl.store(row_sum_ptr + r, acc)


# ---------------------------------------------------------------------------
# Kernel 2: reduce row sums to a single scalar, then divide by rows
# ---------------------------------------------------------------------------
@triton.jit
def reduce_kernel(
    row_sum_ptr,        # [rows]
    scalar_ptr,         # [1]
    rows: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    total = tl.zeros([], dtype=tl.float32)

    # Single iteration since BLOCK_SIZE >= rows (4096 vs 8192? rows=8192)
    # Actually BLOCK_SIZE=4096, so need two iterations. We'll keep loop.
    for start in range(0, rows, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < rows
        vals = tl.load(row_sum_ptr + offsets, mask=mask, other=0.0)
        total += tl.sum(vals)

    # All threads hold the same total; store mean
    tl.store(scalar_ptr, total / rows)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def triton_run(log_p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence with reduction='batchmean'.
    Input shapes: (8192, 8192) both.
    Returns a scalar tensor (0-D).
    """
    assert log_p.shape == q.shape
    rows, cols = log_p.shape

    # Allocate intermediate row sums and final scalar
    row_sum = torch.empty(rows, dtype=torch.float32, device=log_p.device)
    scalar_out = torch.empty(1, dtype=torch.float32, device=log_p.device)

    # Launch parameters for row kernel
    ROWS_PER_PROG = 8          # process 8 rows per program
    BLOCK_SIZE_ROW = cols      # 8192, full row in one iteration
    grid_row = (rows // ROWS_PER_PROG + (rows % ROWS_PER_PROG != 0),)
    row_kl_kernel[grid_row](
        log_p, q, row_sum,
        rows, cols, BLOCK_SIZE_ROW, ROWS_PER_PROG,
        num_warps=8,           # more warps for better latency hiding
    )

    # Launch parameters for reduction kernel
    BLOCK_SIZE_REDUCE = 4096   # reduce loop iterations
    grid_reduce = (1,)
    reduce_kernel[grid_reduce](
        row_sum, scalar_out,
        rows, BLOCK_SIZE_REDUCE,
        num_warps=1,           # single warp, low overhead
    )

    # Return scalar tensor
    return scalar_out[0]