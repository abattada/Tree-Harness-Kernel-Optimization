import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Persistent kernel: each program processes a strided subset of rows,
# computes the KL sum for each row, accumulates the total for all assigned
# rows, and atomically adds it to a global accumulator.
# This fuses the row computation & the reduction, eliminating the
# intermediate row-sum buffer and the separate reduction kernel launch.
# ---------------------------------------------------------------------------
@triton.jit
def kl_div_persistent_kernel(
    log_p_ptr,          # f32 [rows, cols]
    q_ptr,              # f32 [rows, cols]
    global_total_ptr,   # f32 [1]  (atomic accumulator)
    rows: tl.constexpr,
    cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    STRIDE_BLOCKS: tl.constexpr,   # grid size (number of blocks)
):
    pid = tl.program_id(0)
    block_total = tl.zeros([], dtype=tl.float32)

    # Grid-stride loop over rows
    for row_idx in range(pid, rows, STRIDE_BLOCKS):
        row_start = row_idx * cols
        row_sum = tl.zeros([], dtype=tl.float32)

        # Loop over columns in BLOCK_SIZE chunks (exact fit, no mask)
        for col_start in range(0, cols, BLOCK_SIZE):
            offsets = col_start + tl.arange(0, BLOCK_SIZE)
            q_vals = tl.load(
                q_ptr + row_start + offsets,
                eviction_policy='evict_first'
            )
            log_p_vals = tl.load(
                log_p_ptr + row_start + offsets,
                eviction_policy='evict_first'
            )
            # term = q * (log(q) - log_p), safe for q == 0
            term = tl.where(q_vals > 0.0, q_vals * (tl.log(q_vals) - log_p_vals), 0.0)
            row_sum += tl.sum(term)

        block_total += row_sum

    tl.atomic_add(global_total_ptr, block_total)


# ---------------------------------------------------------------------------
# Tiny kernel to scale the total sum by 1/rows  (batchmean)
# ---------------------------------------------------------------------------
@triton.jit
def final_scale_kernel(
    total_ptr,          # f32 [1]
    scalar_ptr,         # f32 [1]
    rows: tl.constexpr,
):
    total = tl.load(total_ptr)
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
    rows, cols = log_p.shape
    assert log_p.shape == q.shape, "Input shapes must match"
    assert log_p.dtype == torch.float32, "Inputs must be float32"

    # Allocate global accumulator and output scalar
    global_total = torch.zeros(1, dtype=torch.float32, device=log_p.device)
    scalar_out = torch.empty(1, dtype=torch.float32, device=log_p.device)

    BLOCK_SIZE: int = 2048          # divides 8192 evenly; 4 loop iterations
    # Use enough blocks to keep the GPU busy, but few enough to keep
    # atomic contention low.  1024 blocks gives ~8 rows/block.
    NUM_BLOCKS: int = min(rows, 1024)
    STRIDE_BLOCKS = NUM_BLOCKS

    # Launch the persistent fusion kernel (≈ all compute + reduction)
    kl_div_persistent_kernel[(NUM_BLOCKS,)](
        log_p, q, global_total,
        rows=rows, cols=cols, BLOCK_SIZE=BLOCK_SIZE,
        STRIDE_BLOCKS=STRIDE_BLOCKS,
        num_warps=16, num_stages=2,
    )

    # Final scaling: divide total by rows to get batchmean
    final_scale_kernel[(1,)](
        global_total, scalar_out, rows=rows,
        num_warps=1,
    )

    # Return a 0‑D scalar tensor
    return scalar_out.squeeze()