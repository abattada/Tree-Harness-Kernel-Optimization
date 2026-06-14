import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fused kernel: compute per-row online softmax + nll loss for a block
# of rows, accumulate a partial sum, and write it out.
# ---------------------------------------------------------------------------
@triton.jit
def fused_loss_kernel(
    logits_ptr,         # f32 [R, N]
    targets_ptr,        # i64 [R]
    partial_sum_ptr,    # f32 [num_partials]
    N: tl.constexpr,               # number of classes
    R: tl.constexpr,               # number of rows
    BLOCK_SIZE: tl.constexpr,      # chunk size along class dim
    ROWS_PER_PROG: tl.constexpr,   # rows handled by one program
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROG
    loss_sum = tl.zeros([], dtype=tl.float32)

    for r in range(ROWS_PER_PROG):
        row_idx = row_start + r
        # Guard against out-of-bounds (last program may have fewer rows)
        if row_idx < R:
            target = tl.load(targets_ptr + row_idx)

            m_old = tl.full([], float('-inf'), dtype=tl.float32)
            d_old = tl.full([], 0.0, dtype=tl.float32)

            row_base = row_idx * N
            for start in range(0, N, BLOCK_SIZE):
                offs = start + tl.arange(0, BLOCK_SIZE)
                mask = offs < N
                x = tl.load(
                    logits_ptr + row_base + offs,
                    mask=mask,
                    other=float('-inf'),
                    eviction_policy='evict_first',
                )
                m_loc = tl.max(x, axis=0)
                m_new = tl.maximum(m_old, m_loc)
                exp_centered = tl.exp(x - m_new)
                sum_exp = tl.sum(exp_centered, axis=0)
                d_new = d_old * tl.exp(m_old - m_new) + sum_exp
                m_old = m_new
                d_old = d_new

            logsumexp = m_old + tl.log(d_old)
            target_logit = tl.load(logits_ptr + row_base + target)
            loss = logsumexp - target_logit
            loss_sum += loss

    tl.store(partial_sum_ptr + pid, loss_sum, eviction_policy='evict_last')


# ---------------------------------------------------------------------------
# Final reduction: sum partials from the fused kernel and divide by R
# to obtain the mean loss.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_mean_kernel(
    partial_sum_ptr,   # f32 [num_partials]
    out_scalar_ptr,    # f32 [1]
    R: tl.constexpr,                # total rows
    num_partials: tl.constexpr,     # number of partial sums
    BLOCK_SIZE: tl.constexpr,       # chosen to cover num_partials
):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < num_partials
    vals = tl.load(partial_sum_ptr + offs, mask=mask, other=0.0,
                   eviction_policy='evict_first')
    total = tl.sum(vals, axis=0)
    mean = total / R
    tl.store(out_scalar_ptr, mean.to(tl.float32))


# ---------------------------------------------------------------------------
# Public entry point – allocates, launches kernels, returns scalar output.
# ---------------------------------------------------------------------------
def triton_run(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute mean cross-entropy loss.

    Args:
        logits:  float32 [8192, 32768]
        targets:  int64  [8192]
    Returns:
        float32 scalar tensor with the mean loss.
    """
    R, N = logits.shape
    assert logits.dtype == torch.float32 and targets.dtype == torch.int64

    # Tuning parameters – chosen for performance on the given shape
    BLOCK_SIZE_CLASS: int = 4096      # chunk along the class dimension
    ROWS_PER_PROG: int = 16           # rows fused into one program
    num_partials = (R + ROWS_PER_PROG - 1) // ROWS_PER_PROG  # ceil div

    partials = torch.empty(num_partials, dtype=torch.float32, device=logits.device)

    # Stage 1: compute per‑block‑of‑rows loss sums
    grid_fused = (num_partials,)
    fused_loss_kernel[grid_fused](
        logits, targets, partials,
        N=N, R=R, BLOCK_SIZE=BLOCK_SIZE_CLASS, ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=8,   # good occupancy for the inner loop
    )

    # Stage 2: sum partials and divide by R
    REDUCE_BLOCK = 512            # comfortably covers num_partials (512 for ROWS=16)
    output = torch.empty(1, dtype=torch.float32, device=logits.device)
    reduce_mean_kernel[(1,)](
        partials, output,
        R=R, num_partials=num_partials, BLOCK_SIZE=REDUCE_BLOCK,
        num_warps=4,   # lightweight reduction
    )

    return output.squeeze()