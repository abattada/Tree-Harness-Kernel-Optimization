import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Persistent kernel: each program processes multiple rows, accumulates a
# partial sum of q*(log q - log_p), then atomically adds it to a global
# scalar.  This fuses the row sum and final reduction into a single kernel,
# eliminates the intermediate row_sum buffer, and drastically reduces the
# number of thread blocks (from 8192 down to ~128) to minimise launch overhead.
# ---------------------------------------------------------------------------
@triton.jit
def fused_kl_kernel(
    log_p_ptr,           # f32 [rows, cols]
    q_ptr,               # f32 [rows, cols]
    global_sum_ptr,      # f32 [1]   — zero-initialised scalar
    rows: tl.constexpr,
    cols: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROG

    # All threads in the block cooperatively load a full row.
    # Each thread gets a contiguous slice of the row, enabling wide vector loads.
    col_offsets = tl.arange(0, cols)
    local_acc = tl.zeros([], dtype=tl.float32)

    for row_offset in range(0, ROWS_PER_PROG):
        row_idx = row_start + row_offset
        if row_idx >= rows:
            break

        row_base = row_idx * cols

        # Coalesced, streaming loads of the whole row with eviction hints
        q_vals = tl.load(
            q_ptr + row_base + col_offsets,
            eviction_policy="evict_first",
        )
        log_p_vals = tl.load(
            log_p_ptr + row_base + col_offsets,
            eviction_policy="evict_first",
        )

        # term = q * (log(q) - log_p)  ;  0 * log(0) → 0
        term = tl.where(q_vals > 0.0,
                        q_vals * (tl.log(q_vals) - log_p_vals),
                        0.0)
        # Accumulate into per-thread partial sum (no inter‑thread sync here)
        local_acc += term

    # Block‑level reduction of the per‑thread partial sums
    block_total = tl.sum(local_acc)

    # Only one thread performs the atomic add to avoid redundant writes
    if tl.thread_id_in_program(0) == 0:
        tl.atomic_add(global_sum_ptr, block_total)


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

    # Zero‑initialised global total
    global_sum = torch.zeros(1, dtype=torch.float32, device=log_p.device)

    # Each program handles ROWS_PER_PROG rows (tunable; 64 gives ≈128 blocks)
    ROWS_PER_PROG = 64
    grid = ((rows + ROWS_PER_PROG - 1) // ROWS_PER_PROG,)

    fused_kl_kernel[grid](
        log_p, q, global_sum,
        rows=rows, cols=cols, ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=8,
    )

    # Return batchmean: total sum divided by batch size (rows)
    return global_sum / rows