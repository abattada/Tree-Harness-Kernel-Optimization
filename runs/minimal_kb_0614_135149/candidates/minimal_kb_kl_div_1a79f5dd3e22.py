import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 1: per-row KL divergence (sum over columns of q*(log(q)-log_p))
# Each program handles one row.
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

    # Loop over columns in BLOCK_SIZE chunks
    for col_start in range(0, cols, BLOCK_SIZE):
        offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < cols

        # Streaming access with cache eviction hint
        q_vals = tl.load(
            q_ptr + row_start + offsets,
            mask=mask,
            other=0.0,
            eviction_policy='evict_first',
        )
        log_p_vals = tl.load(
            log_p_ptr + row_start + offsets,
            mask=mask,
            other=0.0,
            eviction_policy='evict_first',
        )

        # term = q * (log(q) - log_p), safe for q == 0
        term = tl.where(q_vals > 0.0, q_vals * (tl.log(q_vals) - log_p_vals), 0.0)
        acc += tl.sum(term)

    tl.store(row_sum_ptr + pid, acc)


# ---------------------------------------------------------------------------
# Kernel 2: cheap reduction of per-row sums -> scalar mean
# ---------------------------------------------------------------------------
@triton.jit
def reduce_row_sums_kernel(
    row_sum_ptr,        # f32 [rows]
    scalar_ptr,         # f32 [1]
    rows: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    total = tl.zeros([], dtype=tl.float32)
    for start in range(0, rows, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < rows
        vals = tl.load(
            row_sum_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy='evict_first',
        )
        total += tl.sum(vals)  # block-wide reduction

    tl.store(scalar_ptr, total / rows)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def triton_run(log_p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    KL divergence (batchmean) for 2D tensors.
    Input:  log_p (f32[8192,8192]), q (f32[8192,8192])
    Output: scalar f32[] tensor
    """
    assert log_p.shape == q.shape and log_p.ndim == 2, "Expected 2D tensors"
    rows, cols = log_p.shape

    # Intermediate storage for per-row KL sums
    row_sum = torch.empty(rows, dtype=torch.float32, device=log_p.device)
    scalar_out = torch.empty(1, dtype=torch.float32, device=log_p.device)

    # ----- tuneable block sizes -----
    BLOCK_SIZE_ROW = 2048   # divisor of 8192; reduces loops without exceeding SM limits
    BLOCK_SIZE_RED = 1024   # efficient reduction block size

    # Launch row kernel (one program per row)
    grid_row = (rows,)
    row_kl_kernel[grid_row](
        log_p, q, row_sum,
        rows, cols,
        BLOCK_SIZE=BLOCK_SIZE_ROW,
    )

    # Launch scalar reduction (single block)
    reduce_row_sums_kernel[(1,)](
        row_sum, scalar_out,
        rows,
        BLOCK_SIZE=BLOCK_SIZE_RED,
    )

    return scalar_out