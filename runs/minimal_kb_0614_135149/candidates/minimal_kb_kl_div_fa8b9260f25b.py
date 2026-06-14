import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 1: per-row KL divergence sum – each program processes one entire row.
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

    # BLOCK_SIZE == cols -> no loop, no mask required
    offsets = tl.arange(0, BLOCK_SIZE)

    # Coalesced loads of the whole row with eviction hints
    q_vals = tl.load(q_ptr + row_start + offsets,
                     eviction_policy='evict_first')
    log_p_vals = tl.load(log_p_ptr + row_start + offsets,
                         eviction_policy='evict_first')

    # term = q * (log(q) - log_p) ; safely handle q == 0
    term = tl.where(q_vals > 0.0,
                    q_vals * (tl.log(q_vals) - log_p_vals),
                    0.0)

    # Block‑level sum (tree reduction in shared memory)
    acc = tl.sum(term)
    tl.store(row_sum_ptr + pid, acc)


# ---------------------------------------------------------------------------
# Kernel 2: reduce per-row sums to a scalar (batchmean).
#   row_sum: [rows]
#   scalar_out: [1]
# ---------------------------------------------------------------------------
@triton.jit
def reduce_scalar_kernel(
    row_sum_ptr,        # f32 [rows]
    scalar_ptr,         # f32 [1]
    rows: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # BLOCK_SIZE == rows -> single load of all row sums
    offsets = tl.arange(0, BLOCK_SIZE)
    vals = tl.load(row_sum_ptr + offsets,
                   eviction_policy='evict_first')

    total = tl.sum(vals)
    mean = total / rows
    tl.store(scalar_ptr, mean)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def triton_run(log_p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    KL divergence with batchmean reduction:
        sum(q * (log q - log_p)) / batch_size.
    Inputs: both [8192, 8192], float32.
    Returns: scalar float32 tensor.
    """
    rows, cols = log_p.shape
    assert log_p.shape == q.shape, "Input shapes must match"
    assert log_p.dtype == torch.float32, "Inputs must be float32"

    # Allocate intermediate per-row sums and output scalar
    row_sum = torch.empty(rows, dtype=torch.float32, device=log_p.device)
    scalar_out = torch.empty(1, dtype=torch.float32, device=log_p.device)

    # Shapes as compile‑time constants
    ROWS = rows       # 8192
    COLS = cols       # 8192

    # Tuned constants: one block covers the whole row / whole reduction
    BLOCK_COLS = 8192
    BLOCK_ROWS = 8192

    # Row kernel: one program per row, loads the entire row at once
    row_kl_kernel[(ROWS,)](
        log_p, q, row_sum,
        rows=ROWS, cols=COLS, BLOCK_SIZE=BLOCK_COLS,
        num_warps=16
    )

    # Reduction kernel: one program processes all row sums in one shot
    reduce_scalar_kernel[(1,)](
        row_sum, scalar_out,
        rows=ROWS, BLOCK_SIZE=BLOCK_ROWS,
        num_warps=2
    )

    # Return a 0‑D scalar tensor (batchmean)
    return scalar_out.squeeze()