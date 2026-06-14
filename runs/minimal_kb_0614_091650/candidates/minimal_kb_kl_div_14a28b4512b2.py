import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 1: compute per-row KL divergence sum over columns.
# One program per row. Processes columns in BLOCK_SIZE chunks.
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
    for col_start in range(0, cols, BLOCK_SIZE):
        offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < cols
        # Load q and log_p
        q_vals = tl.load(q_ptr + row_start + offsets, mask, other=0.0)
        log_p_vals = tl.load(log_p_ptr + row_start + offsets, mask, other=0.0)
        # term = q * (log(q) - log_p); treat q=0 as term=0 (avoid log(0))
        term = tl.where(q_vals > 0, q_vals * (tl.log(q_vals) - log_p_vals), 0.0)
        acc += tl.sum(term)
    tl.store(row_sum_ptr + pid, acc)


# ---------------------------------------------------------------------------
# Kernel 2: reduce per-row sums into a scalar (batchmean).
# A single program sums all rows and divides by rows.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_kernel(
    row_sum_ptr,        # f32 [rows]
    scalar_ptr,         # f32 [1]
    rows: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    total = tl.zeros([], dtype=tl.float32)
    for start in range(0, rows, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < rows
        vals = tl.load(row_sum_ptr + offsets, mask, other=0.0)
        total += tl.sum(vals)
    # All threads have the same total; store once.
    tl.store(scalar_ptr, total / rows)


# ---------------------------------------------------------------------------
# Public entry point: compute KL divergence (batchmean reduction).
# ---------------------------------------------------------------------------
def triton_run(log_p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    assert log_p.shape == q.shape
    assert log_p.device == q.device
    rows, cols = log_p.shape

    # Allocate temporary row sums and scalar output
    row_sum = torch.empty(rows, dtype=torch.float32, device=log_p.device)
    scalar_out = torch.empty(1, dtype=torch.float32, device=log_p.device)

    BLOCK_SIZE = 1024  # Sensible default for Blackwell; can be tuned.
    grid_row = (rows,)
    row_kl_kernel[grid_row](
        log_p, q, row_sum,
        rows, cols, BLOCK_SIZE,
        num_warps=8,
    )

    grid_red = (1,)
    reduce_kernel[grid_red](
        row_sum, scalar_out,
        rows, BLOCK_SIZE,
        num_warps=8,
    )

    return scalar_out.squeeze(0)