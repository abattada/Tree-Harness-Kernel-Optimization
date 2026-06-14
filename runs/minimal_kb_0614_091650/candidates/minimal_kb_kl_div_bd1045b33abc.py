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
# Kernel 2a: first reduction stage – each block sums a chunk of row sums
# ---------------------------------------------------------------------------
@triton.jit
def reduce_stage1_kernel(
    row_sum_ptr,        # [rows]
    partial_ptr,        # [num_blocks]
    rows: tl.constexpr,
    BLOCK_SIZE_RED: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE_RED
    offs = start + tl.arange(0, BLOCK_SIZE_RED)
    mask = offs < rows
    vals = tl.load(row_sum_ptr + offs, mask=mask, other=0.0)
    partial = tl.sum(vals, axis=0)
    tl.store(partial_ptr + pid, partial)


# ---------------------------------------------------------------------------
# Kernel 2b: second reduction stage – sum the partials and divide by rows
# ---------------------------------------------------------------------------
@triton.jit
def reduce_stage2_kernel(
    partial_ptr,        # [num_blocks]
    scalar_ptr,         # [1]
    num_blocks: tl.constexpr,
    rows: tl.constexpr,
):
    total = tl.zeros([], dtype=tl.float32)
    for start in range(0, num_blocks, 1):
        # Each block writes one value; load it directly
        offs = start
        if offs < num_blocks:
            val = tl.load(partial_ptr + offs)
            total += val
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
    # Output scalar
    scalar_out = torch.empty(1, dtype=torch.float32, device=log_p.device)

    # Tuning knobs
    BLOCK_SIZE = 1024          # for row kernel (columns)
    BLOCK_SIZE_RED = 1024      # for reduction stage 1 (number of rows per block)
    NUM_WARPS = 8

    # Kernel 1: one program per row
    grid_row = (rows,)
    row_kl_kernel[grid_row](
        log_p, q, row_sum,
        rows, cols, BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )

    # Two-stage reduction
    num_stage1_blocks = (rows + BLOCK_SIZE_RED - 1) // BLOCK_SIZE_RED
    partial = torch.empty(num_stage1_blocks, dtype=torch.float32, device=log_p.device)

    grid_s1 = (num_stage1_blocks,)
    reduce_stage1_kernel[grid_s1](
        row_sum, partial,
        rows, BLOCK_SIZE_RED,
        num_warps=NUM_WARPS,
    )

    grid_s2 = (1,)
    reduce_stage2_kernel[grid_s2](
        partial, scalar_out,
        num_stage1_blocks, rows,
        num_warps=NUM_WARPS,
    )

    return scalar_out.squeeze(0)