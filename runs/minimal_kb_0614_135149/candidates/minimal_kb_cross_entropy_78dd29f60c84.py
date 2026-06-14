import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Fast‑math online softmax + NLL loss, one program per row.
# Uses exp2 / log2 for higher throughput on Blackwell.
# ---------------------------------------------------------------------------
@triton.jit
def cross_entropy_row_kernel(
    logits_ptr,      # f32 [R, N]
    targets_ptr,     # i64 [R]
    loss_row_ptr,    # f32 [R]
    N: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Constants for exp2/log2 rescaling
    LOG2E: tl.constexpr = 1.4426950408889634
    LN2:  tl.constexpr = 0.6931471805599453

    pid = tl.program_id(0)
    target = tl.load(targets_ptr + pid)

    m_curr = tl.full([], float('-inf'), dtype=tl.float32)
    d_curr = tl.full([], 0.0, dtype=tl.float32)

    row_base = pid * N
    for start in range(0, N, BLOCK_SIZE_N):
        offs = start + tl.arange(0, BLOCK_SIZE_N)
        x = tl.load(logits_ptr + row_base + offs,
                    eviction_policy='evict_first')

        m_loc = tl.max(x, axis=0)
        m_new = tl.maximum(m_curr, m_loc)

        # exp(x - m_new) via exp2
        shifted = (x - m_new) * LOG2E
        exp_centered = tl.exp2(shifted)
        sum_exp = tl.sum(exp_centered, axis=0)

        d_new = d_curr * tl.exp2((m_curr - m_new) * LOG2E) + sum_exp
        m_curr = m_new
        d_curr = d_new

    # logsumexp = m_curr + ln(d_curr) via log2
    logsumexp = m_curr + LN2 * tl.log2(d_curr)

    target_logit = tl.load(logits_ptr + row_base + target)
    loss = logsumexp - target_logit
    tl.store(loss_row_ptr + pid, loss, eviction_policy='evict_last')


# ---------------------------------------------------------------------------
# First reduction stage – each block sums a chunk of per‑row losses.
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
# Second reduction stage – sums all partials and computes the mean.
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
    logits : f32 [8192, 32768]
    targets: i64 [8192]
    returns: f32 scalar = mean cross-entropy loss
    """
    R, N = logits.shape
    assert targets.shape == (R,)
    assert logits.dtype == torch.float32
    assert targets.dtype == torch.int64

    # Tuning values – all divide the respective dimension exactly
    BLOCK_SIZE_N = 2048
    BLOCK_SIZE_R = 512
    NUM_WARPS_ROW = 8
    NUM_WARPS_R1  = 8
    NUM_WARPS_R2  = 1

    # 1. Per‑row losses
    loss_row = torch.empty(R, dtype=torch.float32, device=logits.device)
    cross_entropy_row_kernel[(R,)](
        logits, targets, loss_row,
        N, BLOCK_SIZE_N,
        num_warps=NUM_WARPS_ROW,
    )

    # 2. First‑stage reduction → partial sums
    num_partials = R // BLOCK_SIZE_R   # 16
    partials = torch.empty(num_partials, dtype=torch.float32, device=logits.device)
    reduce_sum_stage1_kernel[(num_partials,)](
        loss_row, partials,
        R, BLOCK_SIZE_R,
        num_warps=NUM_WARPS_R1,
    )

    # 3. Second‑stage reduction → scalar mean
    out = torch.empty((), dtype=torch.float32, device=logits.device)
    BLOCK_SIZE_R2 = num_partials
    reduce_mean_stage2_kernel[(1,)](
        partials, out,
        R, num_partials, BLOCK_SIZE_R2,
        num_warps=NUM_WARPS_R2,
    )

    return out