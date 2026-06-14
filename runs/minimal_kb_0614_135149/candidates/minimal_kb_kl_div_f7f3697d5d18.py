import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 1: grid-strided row processing – each program handles multiple rows,
# accumulates their KL divergence sums, and writes a partial total.
# ---------------------------------------------------------------------------
@triton.jit
def row_kl_grid_kernel(
    log_p_ptr,          # f32 [rows, cols]
    q_ptr,              # f32 [rows, cols]
    partial_sum_ptr,    # f32 [TOTAL_PROGRAMS]   output
    rows: tl.constexpr,
    cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TOTAL_PROGRAMS: tl.constexpr,
):
    pid = tl.program_id(0)
    acc = tl.zeros([], dtype=tl.float32)

    # grid-stride loop over rows
    row_idx = pid
    while row_idx < rows:
        row_base = row_idx * cols
        # loop over columns in BLOCK_SIZE-sized chunks
        for col_start in range(0, cols, BLOCK_SIZE):
            offsets = col_start + tl.arange(0, BLOCK_SIZE)
            # cols is a multiple of BLOCK_SIZE → no mask needed
            q_vals = tl.load(
                q_ptr + row_base + offsets,
                eviction_policy='evict_first',
            )
            log_p_vals = tl.load(
                log_p_ptr + row_base + offsets,
                eviction_policy='evict_first',
            )
            # term = q * (log(q) - log_p), safe for q == 0
            term = tl.where(
                q_vals > 0.0,
                q_vals * (tl.log(q_vals) - log_p_vals),
                0.0,
            )
            acc += tl.sum(term)
        row_idx += TOTAL_PROGRAMS

    # write this program's partial total
    tl.store(partial_sum_ptr + pid, acc)


# ---------------------------------------------------------------------------
# Kernel 2: reduce the per-program partial sums to a scalar (batchmean)
# ---------------------------------------------------------------------------
@triton.jit
def reduce_partials_kernel(
    partial_sum_ptr,    # f32 [TOTAL_PROGRAMS]
    scalar_ptr,         # f32 [1]
    TOTAL_PROGRAMS: tl.constexpr,
    rows: tl.constexpr,
):
    offsets = tl.arange(0, TOTAL_PROGRAMS)
    vals = tl.load(partial_sum_ptr + offsets)
    # block-wide reduction sum
    total = tl.reduce(vals, axis=0, combine_fn=lambda a, b: a + b)
    tl.store(scalar_ptr, total / rows)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def triton_run(log_p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    KL divergence with batchmean reduction: sum(q*(log q - log_p)) / batch_size.
    Inputs: both [8192, 8192] float32.
    Returns: scalar float32 tensor.
    """
    assert log_p.shape == q.shape, "Input shapes must match"
    assert log_p.dtype == torch.float32, "Inputs must be float32"

    rows, cols = log_p.shape
    # Tunable constants for the 8192 × 8192 problem
    BLOCK_SIZE = 2048          # divides 8192 evenly, good register/occupancy trade-off
    TOTAL_PROGRAMS = 256       # each program processes rows/TOTAL_PROGRAMS = 32 rows

    # Intermediate partial sums
    partial_sums = torch.empty(TOTAL_PROGRAMS, dtype=torch.float32, device=log_p.device)
    scalar_out = torch.empty(1, dtype=torch.float32, device=log_p.device)

    # Row kernel: grid-stride loop over rows
    grid_row = (TOTAL_PROGRAMS,)
    row_kl_grid_kernel[grid_row](
        log_p, q, partial_sums,
        rows=rows, cols=cols,
        BLOCK_SIZE=BLOCK_SIZE,
        TOTAL_PROGRAMS=TOTAL_PROGRAMS,
        num_warps=16,
        num_stages=2,
    )

    # Reduction kernel: single program, parallel reduction of TOTAL_PROGRAMS elements
    reduce_partials_kernel[(1,)](
        partial_sums, scalar_out,
        TOTAL_PROGRAMS=TOTAL_PROGRAMS,
        rows=rows,
        num_warps=8,            # 256 threads, one per partial sum
        num_stages=1,
    )

    return scalar_out.squeeze()