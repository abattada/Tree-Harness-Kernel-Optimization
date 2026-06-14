import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 1: per-row KL divergence, single-pass over columns.
# Both rows and cols are constexpr (specialized at 8192,8192) to allow
# compiler optimizations (mask elimination, vectorization).
# ---------------------------------------------------------------------------
@triton.jit
def row_kl_kernel(
    log_p_ptr,          # [rows, cols]
    q_ptr,              # [rows, cols]
    row_sum_ptr,        # [rows] (output)
    rows: tl.constexpr,
    cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * cols

    acc = tl.zeros([], dtype=tl.float32)

    # Only one iteration because BLOCK_SIZE == cols (boundary mask always true)
    for col_start in range(0, cols, BLOCK_SIZE):
        offsets = col_start + tl.arange(0, BLOCK_SIZE)
        # No mask needed because cols is a multiple of BLOCK_SIZE,
        # but we keep it for generality; compiler will optimize it away.
        mask = offsets < cols

        # Load q and log_p with streaming eviction hints and alignment hints
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

        # term = q * (log(q) - log_p); safe handling for q == 0
        term = tl.where(q_vals > 0, q_vals * (tl.log(q_vals) - log_p_vals), 0.0)
        acc += tl.sum(term)

    tl.store(row_sum_ptr + pid, acc)


# ---------------------------------------------------------------------------
# Kernel 2: reduce row sums to a scalar, divide by rows.
# Uses a single BLOCK of size rows (8192) to finish in one iteration.
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

    tl.store(scalar_ptr, total / rows)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def triton_run(log_p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence with reduction='batchmean'.
    Input shapes: (8192, 8192) for both log_p and q.
    Returns a 0-D scalar tensor.
    """
    rows, cols = log_p.shape
    assert log_p.shape == q.shape
    assert rows == 8192 and cols == 8192  # expected by specialization

    # Allocate intermediate row sums and final scalar output
    row_sum = torch.empty(rows, dtype=torch.float32, device=log_p.device)
    scalar_out = torch.empty(1, dtype=torch.float32, device=log_p.device)

    # Launch row kernel: each program handles one row, BLOCK_SIZE = cols
    grid_row = (rows,)
    row_kl_kernel[grid_row](
        log_p, q, row_sum,
        rows, cols, cols,           # BLOCK_SIZE = cols = 8192
        num_warps=4,
    )

    # Launch reduction kernel: single program, BLOCK_SIZE = rows
    grid_reduce = (1,)
    reduce_row_sums_kernel[grid_reduce](
        row_sum, scalar_out,
        rows, rows,                 # BLOCK_SIZE = rows = 8192
        num_warps=4,
    )

    return scalar_out[0]