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
    row_sum_ptr,        # [rows]
    rows: tl.constexpr,
    cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * cols

    acc = tl.zeros([], dtype=tl.float32)

    # Loop over columns in blocks
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

        # term = q * (log(q) - log_p), safely zero when q == 0
        term = tl.where(q_vals > 0, q_vals * (tl.log(q_vals) - log_p_vals), 0.0)
        acc += tl.sum(term)

    tl.store(row_sum_ptr + pid, acc)


# ---------------------------------------------------------------------------
# Kernel 2: reduction of row sums to a single scalar, then divide by rows
# ---------------------------------------------------------------------------
@triton.jit
def reduce_row_sums_kernel(
    row_sum_ptr,        # [rows]
    scalar_ptr,         # [1] (we later reshape to scalar)
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

    # Store the mean (divide by number of rows)
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
    rows, cols = log_p.shape
    assert q.shape == (rows, cols)

    # Allocate intermediate row sums and final scalar output
    row_sum = torch.empty(rows, dtype=torch.float32, device=log_p.device)
    scalar_out = torch.empty(1, dtype=torch.float32, device=log_p.device)

    # Launch row kernel
    BLOCK_SIZE_ROW = 2048
    grid_row = (rows,)
    row_kl_kernel[grid_row](
        log_p, q, row_sum,
        rows, cols,
        BLOCK_SIZE=BLOCK_SIZE_ROW,
        num_warps=4,
    )

    # Launch reduction kernel
    BLOCK_SIZE_REDUCE = 4096
    grid_reduce = (1,)
    reduce_row_sums_kernel[grid_reduce](
        row_sum, scalar_out,
        rows,
        BLOCK_SIZE=BLOCK_SIZE_REDUCE,
        num_warps=1,
    )

    # Return scalar (0-dim tensor)
    return scalar_out.view(())