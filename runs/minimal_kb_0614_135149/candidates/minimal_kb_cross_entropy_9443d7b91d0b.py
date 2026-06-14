import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: fused per‑program multi‑row online softmax + NLL loss.
# Each program processes ROWS_PER_PROG rows, accumulates their losses into a
# partial sum, and stores that single partial.
# ---------------------------------------------------------------------------
@triton.jit
def fused_cross_entropy_kernel(
    logits_ptr,      # float32 [R, N]
    targets_ptr,      # int64   [R]
    partials_ptr,     # float32 [num_partials]
    R: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROG

    total_loss = tl.zeros([], dtype=tl.float32)

    for r in range(ROWS_PER_PROG):
        row_idx = row_start + r
        if row_idx >= R:
            break                       # past the end of data

        target = tl.load(targets_ptr + row_idx)

        # Online softmax over the row
        m_curr = tl.full([], float('-inf'), dtype=tl.float32)
        d_curr = tl.full([], 0.0, dtype=tl.float32)

        row_base = row_idx * N
        for start in range(0, N, BLOCK_SIZE_N):
            offs = start + tl.arange(0, BLOCK_SIZE_N)
            x = tl.load(
                logits_ptr + row_base + offs,
                eviction_policy='evict_first',
            )
            m_loc = tl.max(x, axis=0)
            m_new = tl.maximum(m_curr, m_loc)
            exp_centered = tl.exp(x - m_new)
            sum_exp = tl.sum(exp_centered, axis=0)
            d_new = d_curr * tl.exp(m_curr - m_new) + sum_exp
            m_curr = m_new
            d_curr = d_new

        logsumexp = m_curr + tl.log(d_curr)
        target_logit = tl.load(logits_ptr + row_base + target)
        loss = logsumexp - target_logit
        total_loss += loss

    tl.store(partials_ptr + pid, total_loss)


# ---------------------------------------------------------------------------
# Kernel 2: single‑block reduction over partials → scalar mean.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_mean_kernel(
    partials_ptr,     # float32 [num_partials]
    out_ptr,          # float32 scalar
    R: tl.constexpr,
    num_partials: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < num_partials
    vals = tl.load(partials_ptr + offs, mask=mask, other=0.0,
                   eviction_policy='evict_first')
    total = tl.sum(vals, axis=0)
    mean = total / R
    tl.store(out_ptr, mean)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def triton_run(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits : float32 [8192, 32768]
    targets: int64   [8192]
    returns: float32 scalar = mean cross-entropy loss
    """
    R, N = logits.shape
    assert targets.shape == (R,)
    assert logits.dtype == torch.float32
    assert targets.dtype == torch.int64

    # Tuning choices – exact divisors of the problem dimensions
    BLOCK_SIZE_N = 4096        # yields 8 iterations per row (vs 16 with 2048)
    ROWS_PER_PROG = 8          # 1024 programs, good occupancy

    NUM_WARPS_FUSED = 8        # 256 threads per block
    NUM_WARPS_REDUCE = 4       # 128 threads for the small reduction

    num_partials = (R + ROWS_PER_PROG - 1) // ROWS_PER_PROG  # = 1024

    # Step 1: fused row processing + first stage reduction
    partials = torch.empty(num_partials, dtype=torch.float32, device=logits.device)
    fused_cross_entropy_kernel[(num_partials,)](
        logits, targets, partials,
        R, N, BLOCK_SIZE_N, ROWS_PER_PROG,
        num_warps=NUM_WARPS_FUSED,
    )

    # Step 2: final sum of partials and division by R
    out = torch.empty((), dtype=torch.float32, device=logits.device)
    # BLOCK_SIZE is large enough to cover all partials in one go
    BLOCK_SIZE_REDUCE = 1024  # == num_partials exactly
    reduce_mean_kernel[(1,)](
        partials, out,
        R, num_partials, BLOCK_SIZE_REDUCE,
        num_warps=NUM_WARPS_REDUCE,
    )

    return out