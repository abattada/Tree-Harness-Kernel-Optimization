import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 1: Multi‑row cross‑entropy with one‑shot logsumexp per row.
# Each program processes ROWS_PER_PROG rows, loads the whole row at once,
# computes logsumexp directly, and accumulates the per‑row loss.
# Writes a partial sum to the output buffer.
# ---------------------------------------------------------------------------
@triton.jit
def cross_entropy_multirow_kernel(
    logits_ptr,         # f32 [R, N]
    targets_ptr,        # i64 [R]
    partial_sum_ptr,    # f32 [num_partials]
    N: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
    R: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROG
    row_end = tl.minimum(row_start + ROWS_PER_PROG, R)

    # Offsets for loading a whole row at once
    offs = tl.arange(0, N)

    accum = tl.zeros([], dtype=tl.float32)
    for row_idx in range(row_start, row_end):
        target = tl.load(targets_ptr + row_idx)
        row_base = row_idx * N

        x = tl.load(logits_ptr + row_base + offs,
                    eviction_policy='evict_first')

        m = tl.max(x, axis=0)
        d = tl.sum(tl.exp(x - m), axis=0)
        logsumexp = m + tl.log(d)

        target_logit = tl.load(logits_ptr + row_base + target)
        loss = logsumexp - target_logit
        accum += loss

    tl.store(partial_sum_ptr + pid, accum)


# ---------------------------------------------------------------------------
# Kernel 2: Final reduction of partial sums into the mean loss.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_mean_kernel(
    partial_sum_ptr,   # f32 [NUM_PARTIALS]
    out_ptr,           # f32 [1]
    R: tl.constexpr,
    NUM_PARTIALS: tl.constexpr,
):
    offs = tl.arange(0, NUM_PARTIALS)
    vals = tl.load(partial_sum_ptr + offs)
    total = tl.sum(vals, axis=0)
    mean = total / R
    tl.store(out_ptr, mean)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def triton_run(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Mean cross‑entropy loss.

    Args:
        logits:  float32 [8192, 32768]
        targets: int64   [8192]

    Returns:
        scalar float32 tensor with mean loss.
    """
    R, N = logits.shape
    assert logits.dtype == torch.float32
    assert targets.dtype == torch.int64

    # --- Tuning constants for RTX 5090, shapes 8192 × 32768 ---
    ROWS_PER_PROG = 32      # each program handles 32 rows
    num_blocks = triton.cdiv(R, ROWS_PER_PROG)   # 256 blocks
    partial_sums = torch.empty(num_blocks, dtype=torch.float32, device=logits.device)

    # Stage 1: per‑block loss accumulation
    cross_entropy_multirow_kernel[(num_blocks,)](
        logits, targets, partial_sums,
        N=N,
        ROWS_PER_PROG=ROWS_PER_PROG,
        R=R,
        num_warps=8, num_stages=2,
    )

    # Stage 2: sum partials and divide by R
    output = torch.empty(1, dtype=torch.float32, device=logits.device)
    reduce_mean_kernel[(1,)](
        partial_sums, output,
        R=R,
        NUM_PARTIALS=num_blocks,
        num_warps=4, num_stages=1,
    )

    return output.squeeze()