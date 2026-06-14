import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 1: per-row KL divergence contribution (sum over classes)
# Each program handles one row, iterating over columns in BLOCK_SIZE chunks.
# ---------------------------------------------------------------------------
@triton.jit
def row_kl_kernel(
    log_p_ptr,          # [rows, cols]
    q_ptr,              # [rows, cols]
    row_sum_ptr,        # [rows]   (output, per-row sum)
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
        q_vals = tl.load(q_ptr + row_start + offsets, mask, other=0.0)
        log_p_vals = tl.load(log_p_ptr + row_start + offsets, mask, other=0.0)
        # term = q * (log(q) - log_p)  ; avoid log(0)*0 -> NaN
        term = tl.where(q_vals > 0, q_vals * (tl.log(q_vals) - log_p_vals), 0.0)
        acc += tl.sum(term)
    tl.store(row_sum_ptr + pid, acc)


# ---------------------------------------------------------------------------
# Kernel 2: reduce per-row sums to a scalar and divide by number of rows
# (batchmean = sum(rows) / rows)
# ---------------------------------------------------------------------------
@triton.jit
def reduce_kernel(
    row_sum_ptr,        # [rows]
    scalar_ptr,         # [1]   (output)
    rows: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    total = tl.zeros([], dtype=tl.float32)
    for start in range(0, rows, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < rows
        vals = tl.load(row_sum_ptr + offsets, mask, other=0.0)
        total += tl.sum(vals)
    # All threads have the same total; store by thread 0 (any thread works)
    tl.store(scalar_ptr, total / rows)


def triton_run(log_p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence with batchmean reduction.
    Inputs:
        log_p: f32[8192, 8192]  (log–probabilities, already in log space)
        q:    f32[8192, 8192]  (target distribution)
    Returns:
        scalar f32 tensor: sum(q * (log(q) - log_p)) / batch_size
    """
    assert log_p.shape == q.shape, "log_p and q must have the same shape"
    rows, cols = log_p.shape

    # Row–wise partial sums
    row_sum = torch.empty(rows, dtype=torch.float32, device=log_p.device)
    # Final scalar output
    scalar_out = torch.empty(1, dtype=torch.float32, device=log_p.device)

    # Tuned for RTX 5090 (Blackwell): 1024 elements per load gives good
    # coalescing and occupancy.  8192 cols / 1024 = 8 iterations per row.
    BLOCK_SIZE = 1024
    num_warps = 8

    # Kernel 1: one program per row
    grid_row = (rows,)
    row_kl_kernel[grid_row](
        log_p, q, row_sum,
        rows, cols, BLOCK_SIZE,
        num_warps=num_warps,
    )

    # Kernel 2: single–stage reduction over rows (rows = 8192, fine)
    grid_red = (1,)
    reduce_kernel[grid_red](
        row_sum, scalar_out,
        rows, BLOCK_SIZE,
        num_warps=num_warps,
    )

    return scalar_out.squeeze(0)