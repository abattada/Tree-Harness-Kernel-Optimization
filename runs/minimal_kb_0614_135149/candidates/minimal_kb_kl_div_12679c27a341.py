import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 1: per-row KL divergence sum – each program processes one entire row.
#   log_p: [rows, cols]   q: [rows, cols]   -> row_sum: [rows]
# ---------------------------------------------------------------------------
@triton.jit
def row_kl_kernel(
    log_p_ptr,          # f32 [rows, cols]
    q_ptr,              # f32 [rows, cols]
    row_sum_ptr,        # f32 [rows]
    rows: tl.constexpr,
    cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * cols

    # BLOCK_SIZE == cols → single coalesced load, no mask needed
    offsets = tl.arange(0, BLOCK_SIZE)

    q_vals = tl.load(q_ptr + row_start + offsets,
                     eviction_policy='evict_first')
    log_p_vals = tl.load(log_p_ptr + row_start + offsets,
                         eviction_policy='evict_first')

    # term = q * (log(q) - log_p), safe for q == 0
    term = tl.where(q_vals > 0.0,
                    q_vals * (tl.log(q_vals) - log_p_vals),
                    0.0)

    row_sum = tl.sum(term)
    tl.store(row_sum_ptr + pid, row_sum)


# ---------------------------------------------------------------------------
# Kernel 2: reduce per-row sums to scalar (batchmean)
#   row_sum: [rows]   ->   scalar_out: [1] (mean)
# ---------------------------------------------------------------------------
@triton.jit
def reduce_scalar_kernel(
    row_sum_ptr,        # f32 [rows]
    scalar_ptr,         # f32 [1]
    rows: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    total = tl.zeros([], dtype=tl.float32)

    # Loop over row sums in blocks of BLOCK_SIZE for better latency hiding
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
    KL divergence with batchmean reduction:
        sum(q * (log q - log_p)) / batch_size.
    Inputs: both (8192, 8192), float32.  Returns scalar float32.
    """
    rows, cols = log_p.shape
    assert log_p.shape == q.shape, "Input shapes must match"
    assert log_p.dtype == torch.float32, "Inputs must be float32"

    # Allocate intermediate row sums and output scalar
    row_sum = torch.empty(rows, dtype=torch.float32, device=log_p.device)
    scalar_out = torch.empty(1, dtype=torch.float32, device=log_p.device)

    # Shapes as compile‑time constants
    ROWS: int = rows          # 8192
    COLS: int = cols          # 8192

    # Tuned block sizes: row kernel uses full row; reduction loops over chunks
    BLOCK_COLS = 8192
    BLOCK_ROWS = 4096

    # Launch row kernel: one program per row
    row_kl_kernel[(ROWS,)](
        log_p, q, row_sum,
        rows=ROWS, cols=COLS, BLOCK_SIZE=BLOCK_COLS,
        num_warps=16
    )

    # Launch reduction kernel: one program, loops over row sums
    reduce_scalar_kernel[(1,)](
        row_sum, scalar_out,
        rows=ROWS, BLOCK_SIZE=BLOCK_ROWS,
        num_warps=4
    )

    return scalar_out.squeeze()