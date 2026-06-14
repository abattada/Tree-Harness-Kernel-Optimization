import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 1: per-row KL divergence with grid-stride loop for fewer launches
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
    num_progs: tl.constexpr,
):
    pid = tl.program_id(0)
    # Grid-stride loop: each program processes ROWS_PER_PROG consecutive rows
    for i in range(ROWS_PER_PROG):
        row_idx = pid + i * num_progs
        if row_idx >= rows:
            break
        row_start = row_idx * cols

        acc = tl.zeros([], dtype=tl.float32)
        # Only one iteration if BLOCK_SIZE == cols (8192)
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
            # term = q * (log(q) - log_p) ; avoid 0 * (-inf)
            term = tl.where(q_vals > 0, q_vals * (tl.log(q_vals) - log_p_vals), 0.0)
            acc += tl.sum(term)

        tl.store(row_sum_ptr + row_idx, acc)


# ---------------------------------------------------------------------------
# Kernel 2: reduce row sums to a single scalar (mean over rows)
# Use multiple warps for better parallelism.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_kernel(
    row_sum_ptr,   # [rows]
    scalar_ptr,    # [1]
    rows: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    total = tl.zeros([], dtype=tl.float32)
    for start in range(0, rows, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < rows
        vals = tl.load(row_sum_ptr + offsets, mask=mask, other=0.0,
                       eviction_policy='evict_first')
        total += tl.sum(vals)
    tl.store(scalar_ptr, total / rows)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def triton_run(log_p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence with reduction='batchmean'.
    Inputs: (8192, 8192) f32 tensors.
    Returns a scalar tensor (0-D).
    """
    assert log_p.shape == q.shape
    rows, cols = log_p.shape

    # Allocate intermediate row sums and final scalar
    row_sum = torch.empty(rows, dtype=torch.float32, device=log_p.device)
    scalar_out = torch.empty(1, dtype=torch.float32, device=log_p.device)

    # Launch row kernel with grid-stride loop: aim for ~256 programs
    num_progs = 256
    ROWS_PER_PROG = (rows + num_progs - 1) // num_progs
    BLOCK_SIZE = cols  # full row per iteration
    grid_row = (num_progs,)
    row_kl_kernel[grid_row](
        log_p, q, row_sum,
        rows, cols, BLOCK_SIZE, ROWS_PER_PROG, num_progs,
        num_warps=8,
    )

    # Reduction kernel: use one program with multiple warps, full array at once
    BLOCK_SIZE_REDUCE = 8192 if rows >= 8192 else triton.next_power_of_2(rows)
    grid_reduce = (1,)
    reduce_kernel[grid_reduce](
        row_sum, scalar_out,
        rows, BLOCK_SIZE_REDUCE,
        num_warps=4,
    )

    return scalar_out[0]