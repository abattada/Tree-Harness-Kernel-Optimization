import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 1: per‑multi‑row KL divergence sum.
# Each program handles ROWS_PER_PROG consecutive rows, writing their sums
# to contiguous output positions.  This cuts grid launch size and amortises
# scheduling overhead while keeping the column loop vectorised.
# ---------------------------------------------------------------------------
@triton.jit
def row_kl_kernel(
    log_p_ptr,          # f32 [rows, cols]
    q_ptr,              # f32 [rows, cols]
    row_sum_ptr,        # f32 [rows]
    rows: tl.constexpr,
    cols: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    # Starting row of this program’s slice
    row_base = pid * ROWS_PER_PROG

    for i in range(0, ROWS_PER_PROG):
        row = row_base + i
        # Pointer to the beginning of this row
        row_start = row * cols

        acc = tl.zeros([], dtype=tl.float32)

        # Loop over columns in vectorized tiles; cols=8192, BLOCK_SIZE=2048
        # gives exactly 4 iterations – no mask needed.
        for col_start in range(0, cols, BLOCK_SIZE):
            offsets = col_start + tl.arange(0, BLOCK_SIZE)

            q_vals = tl.load(
                q_ptr + row_start + offsets,
                eviction_policy='evict_first',
            )
            log_p_vals = tl.load(
                log_p_ptr + row_start + offsets,
                eviction_policy='evict_first',
            )

            # term = q * (log(q) - log_p), with safe handling for q == 0
            term = tl.where(
                q_vals > 0.0,
                q_vals * (tl.log(q_vals) - log_p_vals),
                0.0,
            )
            acc += tl.sum(term)

        tl.store(row_sum_ptr + row, acc)


# ---------------------------------------------------------------------------
# Kernel 2: sum the per‑row results and divide by rows (batchmean).
# ---------------------------------------------------------------------------
@triton.jit
def reduce_scalar_kernel(
    row_sum_ptr,        # f32 [rows]
    scalar_ptr,         # f32 [1]
    rows: tl.constexpr,
    REDUCE_BLOCK: tl.constexpr,
):
    total = tl.zeros([], dtype=tl.float32)
    for start in range(0, rows, REDUCE_BLOCK):
        offsets = start + tl.arange(0, REDUCE_BLOCK)
        vals = tl.load(row_sum_ptr + offsets,
                       eviction_policy='evict_first')
        total += tl.sum(vals)
    tl.store(scalar_ptr, total / rows)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def triton_run(log_p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    rows, cols = log_p.shape
    assert log_p.shape == q.shape, "Input shapes must match"
    assert log_p.dtype == torch.float32, "Inputs must be float32"
    assert rows % 16 == 0 and cols % 2048 == 0, "Shapes must be 8192x8192"

    row_sum = torch.empty(rows, dtype=torch.float32, device=log_p.device)
    scalar_out = torch.empty(1, dtype=torch.float32, device=log_p.device)

    # Tuned for 8192x8192
    BLOCK_SIZE = 2048
    REDUCE_BLOCK = 4096
    ROWS_PER_PROG = 16          # 8192 // 16 = 512 programs
    num_programs = rows // ROWS_PER_PROG

    # Row kernel: fewer, larger programs for better occupancy
    row_kl_kernel[(num_programs,)](
        log_p, q, row_sum,
        rows=rows, cols=cols,
        ROWS_PER_PROG=ROWS_PER_PROG,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=2,
    )

    # Scalar reduction kernel (single program, fully unrolled)
    reduce_scalar_kernel[(1,)](
        row_sum, scalar_out,
        rows=rows, REDUCE_BLOCK=REDUCE_BLOCK,
        num_warps=1,
    )

    return scalar_out.squeeze()