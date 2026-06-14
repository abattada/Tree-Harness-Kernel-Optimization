import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 1: compute partial KL sums over ROWS_PER_PROGRAM rows.
# Each program processes a chunk of rows and accumulates the total.
# Output: an array of size number_of_programs.
# ---------------------------------------------------------------------------
@triton.jit
def kl_div_partial_kernel(
    log_p_ptr,               # f32 [rows, cols]
    q_ptr,                   # f32 [rows, cols]
    partial_sums_ptr,        # f32 [num_programs]  (output)
    rows: tl.constexpr,
    cols: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
):
    pid = tl.program_id(0)
    start_row = pid * ROWS_PER_PROGRAM

    total = tl.zeros([], dtype=tl.float32)

    for r in range(ROWS_PER_PROGRAM):
        row_idx = start_row + r
        if row_idx < rows:
            row_base = row_idx * cols
            acc = tl.zeros([], dtype=tl.float32)
            for col_start in range(0, cols, BLOCK_SIZE_COL):
                offsets = col_start + tl.arange(0, BLOCK_SIZE_COL)
                mask = offsets < cols
                # Load q and log_p with eviction policy for streaming
                q_vals = tl.load(
                    q_ptr + row_base + offsets,
                    mask=mask,
                    other=0.0,
                    eviction_policy='evict_first',
                )
                log_p_vals = tl.load(
                    log_p_ptr + row_base + offsets,
                    mask=mask,
                    other=0.0,
                    eviction_policy='evict_first',
                )
                # term = q * (log(q) - log_p). Avoid 0 * (-inf) = NaN.
                term = tl.where(q_vals > 0, q_vals * (tl.log(q_vals) - log_p_vals), 0.0)
                acc += tl.sum(term)
            total += acc

    # Write the partial sum for this program
    tl.store(partial_sums_ptr + pid, total)


# ---------------------------------------------------------------------------
# Kernel 2: reduce the partial sums to a single scalar and divide by rows.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_partial_kernel(
    partial_sums_ptr,        # f32 [N]
    scalar_ptr,              # f32 [1]
    N: tl.constexpr,
    rows: tl.constexpr,      # total rows for division
    BLOCK_SIZE_REDUCE: tl.constexpr,
):
    total = tl.zeros([], dtype=tl.float32)
    for start in range(0, N, BLOCK_SIZE_REDUCE):
        offsets = start + tl.arange(0, BLOCK_SIZE_REDUCE)
        mask = offsets < N
        vals = tl.load(
            partial_sums_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy='evict_first',
        )
        total += tl.sum(vals)
    # Store mean (divide by number of rows)
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

    # Configuration
    ROWS_PER_PROGRAM = 8                    # process 8 rows per program
    BLOCK_SIZE_COL = 8192                   # one column block per row (full row)
    num_programs = (rows + ROWS_PER_PROGRAM - 1) // ROWS_PER_PROGRAM
    BLOCK_SIZE_REDUCE = 1024                # cover all partial sums in one iteration

    # Allocate intermediate buffers
    partial_sums = torch.empty(num_programs, dtype=torch.float32, device=log_p.device)
    scalar_out = torch.empty(1, dtype=torch.float32, device=log_p.device)

    # First kernel: compute per-program partial sums
    grid_partial = (num_programs,)
    kl_div_partial_kernel[grid_partial](
        log_p, q, partial_sums,
        rows, cols,
        ROWS_PER_PROGRAM, BLOCK_SIZE_COL,
        num_warps=4,
    )

    # Second kernel: reduce partial sums to scalar
    grid_reduce = (1,)
    reduce_partial_kernel[grid_reduce](
        partial_sums, scalar_out,
        num_programs, rows,
        BLOCK_SIZE_REDUCE,
        num_warps=4,
    )

    return scalar_out[0]