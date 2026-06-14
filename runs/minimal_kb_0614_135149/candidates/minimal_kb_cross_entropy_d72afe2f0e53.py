import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Persistent kernel: one program per SM (128 total).
# Each program loops over its share of rows with grid‑stride,
# computes per‑row online softmax + NLL loss, and accumulates a partial sum.
# No intermediate per‑row loss array is written.
# ---------------------------------------------------------------------------
@triton.jit
def cross_entropy_persistent(
    logits_ptr,      # float32 [R, N]
    targets_ptr,     # int64   [R]
    partials_ptr,    # float32 [PROGRAMS]
    N: tl.constexpr,           # number of classes (32768)
    BLOCK_N: tl.constexpr,     # tile size along N (divides N)
    R: tl.constexpr,           # number of rows (8192)
    PROGRAMS: tl.constexpr,    # number of SMs (128)
):
    pid = tl.program_id(0)
    sum_loss = tl.zeros([], dtype=tl.float32)

    for row in range(pid, R, PROGRAMS):
        target = tl.load(targets_ptr + row)

        # Online softmax for this row
        m_old = tl.full([], float('-inf'), dtype=tl.float32)
        d_old = tl.full([], 0.0, dtype=tl.float32)

        row_base = row * N
        for start in range(0, N, BLOCK_N):
            offs = start + tl.arange(0, BLOCK_N)
            # N is divisible by BLOCK_N -> no mask needed
            x = tl.load(
                logits_ptr + row_base + offs,
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
        sum_loss += loss

    tl.store(partials_ptr + pid, sum_loss, eviction_policy='evict_last')


# ---------------------------------------------------------------------------
# Second‑stage reduction: sum all partials and divide by R to get the mean.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_mean_final(
    partials_ptr,    # float32 [PROGRAMS]
    out_scalar_ptr,  # float32 []
    R: tl.constexpr,
    PROGRAMS: tl.constexpr,
    BLOCK_R2: tl.constexpr,   # exactly PROGRAMS (128)
):
    offs = tl.arange(0, BLOCK_R2)
    vals = tl.load(partials_ptr + offs)
    total = tl.sum(vals, axis=0)
    mean = total / R
    tl.store(out_scalar_ptr, mean)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def triton_run(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits : float32 [8192, 32768]
    targets: int64   [8192]
    returns: float32 scalar = mean cross‑entropy loss
    """
    R, N = logits.shape
    assert targets.shape == (R,)
    assert logits.dtype == torch.float32
    assert targets.dtype == torch.int64

    BLOCK_N = 2048          # divides N exactly
    PROGRAMS = 128          # number of SMs / persistent programs

    # Persistent first stage: each program writes one partial sum
    partials = torch.empty(PROGRAMS, dtype=torch.float32, device=logits.device)
    cross_entropy_persistent[(PROGRAMS,)](
        logits, targets, partials,
        N, BLOCK_N, R, PROGRAMS,
        num_warps=8,
    )

    # Final reduction: sum partials and compute mean
    out = torch.empty((), dtype=torch.float32, device=logits.device)
    BLOCK_R2 = PROGRAMS
    reduce_mean_final[(1,)](
        partials, out,
        R, PROGRAMS, BLOCK_R2,
        num_warps=2,
    )

    return out