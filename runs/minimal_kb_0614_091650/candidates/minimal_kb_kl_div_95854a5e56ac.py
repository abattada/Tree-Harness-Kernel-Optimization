import torch
import triton
import triton.language as tl

# -----------------------------------------------------------------------
# Kernel 1: process multiple rows per program, accumulating each row's
# KL divergence term and producing a partial sum per program.
# -----------------------------------------------------------------------
@triton.jit
def kl_div_multirow_kernel(
    log_p_ptr,        # [rows, cols] f32
    q_ptr,            # [rows, cols] f32
    partial_ptr,      # [grid_size] f32 output
    cols: tl.constexpr,
    rows_per_prog: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start_row = pid * rows_per_prog
    # Register accumulator for this program's partial sum
    partial_sum = tl.zeros([], dtype=tl.float32)

    for row_offset in range(rows_per_prog):
        row = start_row + row_offset
        row_start = row * cols
        acc = tl.zeros([], dtype=tl.float32)

        for col_start in range(0, cols, BLOCK_SIZE):
            offsets = col_start + tl.arange(0, BLOCK_SIZE)
            # Exact division => no mask needed
            q_vals = tl.load(q_ptr + row_start + offsets,
                             eviction_policy='evict_first')
            log_p_vals = tl.load(log_p_ptr + row_start + offsets,
                                 eviction_policy='evict_first')
            # Compute q * (log(q) - log_p) ; safe handling of q=0
            log_q = tl.where(q_vals > 0, tl.log(q_vals), 0.0)
            term = q_vals * (log_q - log_p_vals)
            acc += tl.sum(term)

        partial_sum += acc

    tl.store(partial_ptr + pid, partial_sum)


# -----------------------------------------------------------------------
# Kernel 2: reduce the (small) list of partial sums to a scalar.
# Because grid_size = rows // rows_per_prog is small (e.g. 64), we do
# everything in a single block with enough threads.
# -----------------------------------------------------------------------
@triton.jit
def reduce_partial_kernel(
    partial_ptr,      # [grid_size] f32
    scalar_ptr,       # [1] f32 output
    grid_size: tl.constexpr,
    rows: tl.constexpr,
):
    offsets = tl.arange(0, grid_size)
    mask = offsets < grid_size
    vals = tl.load(partial_ptr + offsets, mask, other=0.0)
    total = tl.sum(vals)
    # all threads have the same total; store once
    tl.store(scalar_ptr, total / rows)


# -----------------------------------------------------------------------
# Public API: allocate outputs and launch the two kernels.
# -----------------------------------------------------------------------
def triton_run(log_p, q) -> torch.Tensor:
    rows, cols = log_p.shape
    assert q.shape == (rows, cols)
    device = log_p.device

    # Tunable parameters
    BLOCK_SIZE = 2048            # must divide cols
    ROWS_PER_PROG = 128          # must divide rows
    assert cols % BLOCK_SIZE == 0
    assert rows % ROWS_PER_PROG == 0

    grid_size = rows // ROWS_PER_PROG

    partial_sums = torch.empty(grid_size, dtype=torch.float32, device=device)
    scalar_out = torch.empty(1, dtype=torch.float32, device=device)

    # Kernel 1: multi-row processing
    grid = (grid_size,)
    kl_div_multirow_kernel[grid](
        log_p, q, partial_sums,
        cols, ROWS_PER_PROG, BLOCK_SIZE,
        num_warps=8,
    )

    # Kernel 2: final reduction (single block of grid_size threads)
    reduce_partial_kernel[(1,)](
        partial_sums, scalar_out,
        grid_size, rows,
        num_warps=2,       # 64 threads = 2 warps
    )

    return scalar_out.squeeze(0)