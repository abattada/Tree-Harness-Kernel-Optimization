import torch
import triton
import triton.language as tl
import math

# ---------------------------------------------------------------------------
# Fast math constants for exp2/log2 based softmax
# ---------------------------------------------------------------------------
LOG2E = math.log2(math.e)   # 1.4426950408889634
LN2  = math.log(2)          # 0.6931471805599453

# ---------------------------------------------------------------------------
# Kernel 1: per‑row online softmax + nll loss using exp2/log2, one program per row.
# Each row is processed in BLOCK_SIZE chunks along the class dimension.
# ---------------------------------------------------------------------------
@triton.jit
def cross_entropy_online_kernel(
    logits_ptr,      # f32 [R, N]
    targets_ptr,     # i64 [R]
    loss_row_ptr,    # f32 [R]   per‑row loss, to be reduced later
    N: tl.constexpr,           # number of classes
    BLOCK_SIZE: tl.constexpr,  # chunk size along class dimension
):
    pid = tl.program_id(0)
    target = tl.load(targets_ptr + pid)

    m_old = tl.full([], float('-inf'), dtype=tl.float32)
    d_old = tl.full([], 0.0, dtype=tl.float32)

    row_base = pid * N

    # BLOCK_SIZE divides N exactly, so the mask is always True (compiler can eliminate it)
    tl.static_assert(N % BLOCK_SIZE == 0)

    for start in range(0, N, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N   # constant True, but kept for safety
        x = tl.load(
            logits_ptr + row_base + offs,
            mask=mask,
            other=float('-inf'),
            eviction_policy='evict_first',
        )

        m_loc = tl.max(x, axis=0)
        m_new = tl.maximum(m_old, m_loc)

        # Online softmax using exp2/log2
        shifted = (x - m_new) * LOG2E
        exp2_vals = tl.exp2(shifted)
        sum_exp2 = tl.sum(exp2_vals, axis=0)

        # d_new = d_old * exp(m_old - m_new) + sum_exp2_vals
        scale = tl.exp2((m_old - m_new) * LOG2E)
        d_new = d_old * scale + sum_exp2

        m_old = m_new
        d_old = d_new

    # logsumexp = m + ln(d)  where ln(d) = log2(d) * ln(2)
    logsumexp = m_old + tl.log2(d_old) * LN2

    target_logit = tl.load(logits_ptr + row_base + target)
    loss = logsumexp - target_logit
    tl.store(loss_row_ptr + pid, loss, eviction_policy='evict_last')


# ---------------------------------------------------------------------------
# Kernel 2a: first reduction stage – each block sums a chunk of per‑row losses
#            and writes a partial sum.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_sum_stage1_kernel(
    inp_ptr,          # f32 [R]
    out_partial_ptr,  # f32 [num_partials]
    R: tl.constexpr,  # total rows
    BLOCK_SIZE: tl.constexpr,  # rows per program (divides R)
):
    pid = tl.program_id(0)
    tl.static_assert(R % BLOCK_SIZE == 0)
    start = pid * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < R   # always True
    vals = tl.load(
        inp_ptr + offs,
        mask=mask,
        other=0.0,
        eviction_policy='evict_first',
    )
    partial = tl.sum(vals, axis=0)
    tl.store(out_partial_ptr + pid, partial)


# ---------------------------------------------------------------------------
# Kernel 2b: second reduction stage – sums all partials and computes the mean.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_mean_stage2_kernel(
    inp_ptr,          # f32 [num_partials]
    out_scalar_ptr,   # f32 [1]
    R: tl.constexpr,           # total rows (for division)
    num_partials: tl.constexpr,  # number of partial sums
    BLOCK_SIZE: tl.constexpr,    # block size for this reduction (covers all partials)
):
    total = tl.zeros([], dtype=tl.float32)
    tl.static_assert(num_partials % BLOCK_SIZE == 0)
    for start in range(0, num_partials, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < num_partials   # always True
        vals = tl.load(
            inp_ptr + offs,
            mask=mask,
            other=0.0,
            eviction_policy='evict_first',
        )
        total += tl.sum(vals, axis=0)
    mean = total / R
    tl.store(out_scalar_ptr, mean, eviction_policy='evict_last')


# ---------------------------------------------------------------------------
# Main entry point – allocates buffers, launches kernels, returns scalar loss.
# ---------------------------------------------------------------------------
def triton_run(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute mean cross-entropy loss.

    Args:
        logits:  float32 tensor of shape [R, N]  (8192, 32768)
        targets: int64 tensor of shape [R]        (8192,)

    Returns:
        scalar float32 tensor with the mean loss.
    """
    R, N = logits.shape
    assert logits.dtype == torch.float32 and targets.dtype == torch.int64

    # Block sizes tuned for shape [8192, 32768] on sm_120
    BLOCK_SIZE_CLASS = 4096   # divides N=32768 exactly -> 8 chunks per row
    BLOCK_SIZE_REDUCE = 1024  # divides R=8192 exactly -> 8 partial sums
    BLOCK_SIZE_FINAL = 8      # covers all partials in a single iteration

    loss_row = torch.empty(R, dtype=torch.float32, device=logits.device)

    # Row kernel with 8 warps (256 threads) and 3-stage pipelining
    grid_per_row = (R,)
    cross_entropy_online_kernel[grid_per_row](
        logits, targets, loss_row,
        N=N, BLOCK_SIZE=BLOCK_SIZE_CLASS,
        num_warps=8, num_stages=3,
    )

    # Stage 1 reduction: 8 programs, each sums 1024 rows
    num_partials = R // BLOCK_SIZE_REDUCE  # exactly 8
    partials = torch.empty(num_partials, dtype=torch.float32, device=logits.device)
    grid_stage1 = (num_partials,)
    reduce_sum_stage1_kernel[grid_stage1](
        loss_row, partials,
        R=R, BLOCK_SIZE=BLOCK_SIZE_REDUCE,
        num_warps=4, num_stages=1,
    )

    # Stage 2 reduction: final sum and division by R
    output = torch.empty(1, dtype=torch.float32, device=logits.device)
    reduce_mean_stage2_kernel[(1,)](
        partials, output,
        R=R, num_partials=num_partials, BLOCK_SIZE=BLOCK_SIZE_FINAL,
        num_warps=2, num_stages=1,
    )

    return output.squeeze()