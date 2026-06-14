import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 1: per-row KL divergence sum
#   log_p: shape [rows, cols]
#   q:     shape [rows, cols]
#   row_sum: output per-row sums [rows]
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

    acc = tl.zeros([], dtype=tl.float32)

    # Loop over columns; with 8192 cols and BLOCK_SIZE=2048 we have exactly 4 iterations.
    for col_start in range(0, cols, BLOCK_SIZE):
        offsets = col_start + tl.arange(0, BLOCK_SIZE)
        # BLOCK_SIZE divides cols -> no mask needed
        q_vals = tl.load(q_ptr + row_start + offsets,
                         eviction_policy='evict_first')
        log_p_vals = tl.load(log_p_ptr + row_start + offsets,
                             eviction_policy='evict_first')

        # term = q * (log(q) - log_p), safely handling q == 0
        term = tl.where(q_vals > 0.0,
                        q_vals * (tl.log(q_vals) - log_p_vals), 0.0)
        acc += tl.sum(term)

    tl.store(row_sum_ptr + pid, acc)


# ---------------------------------------------------------------------------
# Kernel 2: reduce per-row sums to a scalar (batchmean)
#   row_sum: [rows]
#   scalar_out: [1]  (will be squeezed to scalar)
# ---------------------------------------------------------------------------
@triton.jit
def reduce_scalar_kernel(
    row_sum_ptr,        # f32 [rows]
    scalar_ptr,         # f32 [1]
    rows: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    total = tl.zeros([], dtype=tl.float32)
    for start in range(0, rows, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        # BLOCK_SIZE divides rows -> no mask needed
        vals = tl.load(row_sum_ptr + offsets,
                       eviction_policy='evict_first')
        total += tl.sum(vals)

    mean = total / rows
    tl.store(scalar_ptr, mean)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def triton_run(log_p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    KL divergence with batchmean reduction: sum(q*(log q - log_p)) / batch_size.
    Inputs: both [8192, 8192] float32.
    Returns: scalar float32 tensor.
    """
    rows, cols = log_p.shape
    assert log_p.shape == q.shape, "Input shapes must match"
    assert log_p.dtype == torch.float32, "Inputs must be float32"

    # Intermediate per-row sums
    row_sum = torch.empty(rows, dtype=torch.float32, device=log_p.device)
    scalar_out = torch.empty(1, dtype=torch.float32, device=log_p.device)

    ROWS = rows
    COLS = cols
    # Tuned constants for shapes 8192x8192
    BLOCK_SIZE = 2048       # divides 8192 evenly, good balance of occupancy vs registers
    REDUCE_BLOCK = 4096     # same for the row-reduction pass

    # Row kernel: one program per row
    row_kl_kernel[(ROWS,)](log_p, q, row_sum,
                           rows=ROWS, cols=COLS, BLOCK_SIZE=BLOCK_SIZE,
                           num_warps=16, num_stages=2)

    # Reduction kernel: single program loops over all rows
    reduce_scalar_kernel[(1,)](row_sum, scalar_out,
                               rows=ROWS, BLOCK_SIZE=REDUCE_BLOCK,
                               num_warps=1)

    # Return a 0‑D scalar tensor (batchmean)
    return scalar_out.squeeze()