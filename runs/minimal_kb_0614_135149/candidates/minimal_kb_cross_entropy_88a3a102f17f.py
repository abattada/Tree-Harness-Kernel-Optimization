import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: Persistent kernel – each program processes multiple rows with
#           online softmax, accumulates the per-row losses, and writes a
#           partial sum.  Eliminates the large loss_row intermediate buffer.
# ---------------------------------------------------------------------------
@triton.jit
def cross_entropy_persistent_kernel(
    logits_ptr,       # float32 [R, N]
    targets_ptr,      # int64   [R]
    partials_ptr,     # float32 [NUM_PROGS]
    R: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_PROGS: tl.constexpr,
):
    pid = tl.program_id(0)
    acc_loss = tl.zeros([], dtype=tl.float32)

    for row_idx in range(pid, R, NUM_PROGS):
        target = tl.load(targets_ptr + row_idx)

        # Online softmax per row
        m_curr = tl.full([], float('-inf'), dtype=tl.float32)
        d_curr = tl.full([], 0.0, dtype=tl.float32)
        base = row_idx * N

        for start in range(0, N, BLOCK_N):
            offs = start + tl.arange(0, BLOCK_N)
            x = tl.load(logits_ptr + base + offs, eviction_policy='evict_first')

            m_loc = tl.max(x, axis=0)
            m_new = tl.maximum(m_curr, m_loc)
            exp_centered = tl.exp(x - m_new)
            sum_exp = tl.sum(exp_centered, axis=0)

            d_curr = d_curr * tl.exp(m_curr - m_new) + sum_exp
            m_curr = m_new

        logsumexp = m_curr + tl.log(d_curr)
        target_logit = tl.load(logits_ptr + base + target)
        acc_loss += (logsumexp - target_logit)

    tl.store(partials_ptr + pid, acc_loss)


# ---------------------------------------------------------------------------
# Kernel 2: Sum all partials and divide by R to obtain the mean loss.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_mean_kernel(
    partials_ptr,     # float32 [NUM_PARTIALS]
    out_ptr,          # float32 [1]
    R: tl.constexpr,
    NUM_PARTIALS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    vals = tl.load(partials_ptr + offs)
    total = tl.sum(vals, axis=0)
    mean = total / R
    tl.store(out_ptr, mean)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def triton_run(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    R, N = logits.shape
    assert targets.shape == (R,)
    assert logits.dtype == torch.float32
    assert targets.dtype == torch.int64

    # Tuning choices – all divide their respective dimensions exactly
    BLOCK_N = 2048
    NUM_PROGS = 256
    NUM_WARPS_MAIN = 8

    # Allocate partial sums
    partials = torch.empty(NUM_PROGS, dtype=torch.float32, device=logits.device)

    # Launch persistent kernel
    cross_entropy_persistent_kernel[(NUM_PROGS,)](
        logits, targets, partials,
        R, N, BLOCK_N, NUM_PROGS,
        num_warps=NUM_WARPS_MAIN,
    )

    # Allocate output scalar
    out = torch.empty((), dtype=torch.float32, device=logits.device)

    # Second-stage reduction
    BLOCK_SIZE_R = NUM_PROGS            # exactly 256
    NUM_WARPS_R = (BLOCK_SIZE_R + 31) // 32  # 8 warps
    reduce_mean_kernel[(1,)](
        partials, out,
        R, NUM_PROGS, BLOCK_SIZE_R,
        num_warps=NUM_WARPS_R,
    )

    return out