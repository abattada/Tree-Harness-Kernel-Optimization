import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 1: compute partial sums of KL divergence over groups of rows.
# Each program processes ROWS_PER_PROG consecutive rows and accumulates a
# single partial sum for the entire group.  Inputs are streamed, so we use
# evict_first to avoid polluting the L2 cache.
# ---------------------------------------------------------------------------
@triton.jit
def kl_div_group_kernel(
    log_p_ptr,          # f32 [rows, cols]
    q_ptr,              # f32 [rows, cols]
    partial_sum_ptr,    # f32 [grid_size]   – one partial sum per program
    rows: tl.constexpr,
    cols: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    # First row handled by this program
    row_start = pid * ROWS_PER_PROG
    total = tl.zeros([], dtype=tl.float32)

    for r in range(ROWS_PER_PROG):
        row_idx = row_start + r
        # Guard against leftover rows (if rows not exactly divisible)
        if row_idx < rows:
            acc = tl.zeros([], dtype=tl.float32)
            row_base = row_idx * cols
            # Loop over columns in blocks of BLOCK_SIZE
            for col_start in range(0, cols, BLOCK_SIZE):
                offs = col_start + tl.arange(0, BLOCK_SIZE)
                mask = offs < cols
                q_vals = tl.load(q_ptr + row_base + offs, mask, other=0.0,
                                 eviction_policy='evict_first')
                log_p_vals = tl.load(log_p_ptr + row_base + offs, mask, other=0.0,
                                     eviction_policy='evict_first')
                # term = q * (log(q) - log_p), with zero where q == 0 avoiding NaN
                term = tl.where(q_vals > 0, q_vals * (tl.log(q_vals) - log_p_vals), 0.0)
                acc += tl.sum(term)
            total += acc

    # Write the partial sum for this group of rows
    tl.store(partial_sum_ptr + pid, total)


# ---------------------------------------------------------------------------
# Kernel 2: reduce the partial sums to a scalar, then divide by rows.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_partials_kernel(
    partial_sum_ptr,   # f32 [grid_size]
    scalar_ptr,        # f32 [1]
    grid_size: tl.constexpr,
    rows_total: tl.constexpr,           # total number of rows (for division)
    BLOCK_SIZE: tl.constexpr = 1024,
):
    tid = tl.program_id(0)   # only one program, but we keep signature uniform
    total = tl.zeros([], dtype=tl.float32)
    for start in range(0, grid_size, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < grid_size
        vals = tl.load(partial_sum_ptr + offs, mask, other=0.0)
        total += tl.sum(vals)
    # All threads hold the same total; store once (all threads write the same value)
    tl.store(scalar_ptr, total / rows_total)


# ---------------------------------------------------------------------------
# Wrapper: allocate outputs, launch the two kernels, return the scalar.
# ---------------------------------------------------------------------------
def triton_run(log_p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    assert log_p.shape == q.shape
    rows, cols = log_p.shape

    # Tuned constants
    ROWS_PER_PROG = 4          # each program processes 4 rows
    BLOCK_SIZE = 2048          # columns per iteration (cols=8192, so 4 iterations)

    grid_size = (rows + ROWS_PER_PROG - 1) // ROWS_PER_PROG

    partial_sums = torch.empty(grid_size, dtype=torch.float32, device=log_p.device)
    scalar_out = torch.empty(1, dtype=torch.float32, device=log_p.device)

    # Kernel 1: per‑group KL divergence
    grid = (grid_size,)
    kl_div_group_kernel[grid](
        log_p, q, partial_sums,
        rows, cols, ROWS_PER_PROG, BLOCK_SIZE,
        num_warps=8,
    )

    # Kernel 2: final reduction
    # Use a small block size for the reduce kernel (grid_size is at most 2048)
    reduce_block = 1024 if grid_size >= 1024 else 512
    grid_red = (1,)
    reduce_partials_kernel[grid_red](
        partial_sums, scalar_out,
        grid_size, rows, reduce_block,
        num_warps=4,
    )

    return scalar_out.squeeze(0)