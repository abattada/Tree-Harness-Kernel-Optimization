import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: per‑row online softmax + NLL loss, one program per row.
# Each row is processed in BLOCK_SIZE_N chunks along the class dimension.
# Both N and BLOCK_SIZE_N are known to divide, so boundary masks are omitted.
# ---------------------------------------------------------------------------
@triton.jit
def cross_entropy_row_kernel(
    logits_ptr,      # float32 [R, N]
    targets_ptr,     # int64   [R]
    loss_row_ptr,    # float32 [R]
    N: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    target = tl.load(targets_ptr + pid)

    # Online softmax: compute max and log-sum-exp in a single pass
    m_curr = tl.full([], float('-inf'), dtype=tl.float32)
    d_curr = tl.full([], 0.0, dtype=tl.float32)

    row_base = pid * N
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
    tl.store(loss_row_ptr + pid, loss, eviction_policy='evict_last')


# ---------------------------------------------------------------------------
# Kernel 2a: first reduction stage – each block sums a chunk of per‑row
#            losses and writes a partial sum. R is divisible by BLOCK_SIZE_R.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_sum_stage1_kernel(
    inp_ptr,
    out_partial_ptr,
    R: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE_R
    offs = start + tl.arange(0, BLOCK_SIZE_R)
    vals = tl.load(inp_ptr + offs, eviction_policy='evict_first')
    partial = tl.sum(vals, axis=0)
    tl.store(out_partial_ptr + pid, partial)


# ---------------------------------------------------------------------------
# Kernel 2b: second reduction stage – sums all partials and computes the mean.
#            Launched with exactly num_partials threads, no mask needed.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_mean_stage2_kernel(
    inp_ptr,
    out_scalar_ptr,
    R: tl.constexpr,
    num_partials: tl.constexpr,
    BLOCK_SIZE_R2: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE_R2)
    partials = tl.load(inp_ptr + offs)
    total = tl.sum(partials, axis=0)
    mean = total / R
    tl.store(out_scalar_ptr, mean)


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

    # Tuning choices – all divide the respective dimension exactly
    BLOCK_SIZE_N = 2048
    BLOCK_SIZE_R = 512

    NUM_WARPS_ROW = 8
    NUM_WARPS_R1  = 8
    NUM_WARPS_R2  = 1          # only one warp needed for the tiny final sum

    # Step 1: per-row losses
    loss_row = torch.empty(R, dtype=torch.float32, device=logits.device)
    cross_entropy_row_kernel[(R,)](
        logits, targets, loss_row,
        N, BLOCK_SIZE_N,
        num_warps=NUM_WARPS_ROW,
    )

    # Step 2: first-stage reduction -> partial sums
    num_partials = R // BLOCK_SIZE_R   # 8192 // 512 = 16
    partials = torch.empty(num_partials, dtype=torch.float32, device=logits.device)
    reduce_sum_stage1_kernel[(num_partials,)](
        loss_row, partials,
        R, BLOCK_SIZE_R,
        num_warps=NUM_WARPS_R1,
    )

    # Step 3: second-stage reduction -> scalar mean
    out = torch.empty((), dtype=torch.float32, device=logits.device)
    BLOCK_SIZE_R2 = num_partials    # exactly the number of partials
    reduce_mean_stage2_kernel[(1,)](
        partials, out,
        R, num_partials, BLOCK_SIZE_R2,
        num_warps=NUM_WARPS_R2,
    )

    return out