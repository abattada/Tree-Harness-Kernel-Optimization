import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: persistent row-processing + partial sum accumulation.
# Each program loops over multiple rows (grid‑stride) and adds the per‑row
# cross‑entropy loss to a local sum.  The result is stored in `partials`.
# ---------------------------------------------------------------------------
@triton.jit
def cross_entropy_persistent_kernel(
    logits_ptr,      # float32 [R, N]
    targets_ptr,     # int64   [R]
    partials_ptr,    # float32 [NUM_PROGRAMS]
    R: tl.constexpr,
    N: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    local_sum = tl.zeros([], dtype=tl.float32)

    # Grid‑stride loop over rows: each program owns `pid + k*NUM_PROGRAMS`
    for row in range(pid, R, NUM_PROGRAMS):
        target = tl.load(targets_ptr + row)

        # Online softmax for a single row (BLOCK_SIZE_N divides N exactly)
        m_curr = tl.full([], float('-inf'), dtype=tl.float32)
        d_curr = tl.full([], 0.0, dtype=tl.float32)

        row_base = row * N
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
        local_sum += loss

    tl.store(partials_ptr + pid, local_sum)


# ---------------------------------------------------------------------------
# Kernel 2: final reduction – sum all partials and divide by R to get the
#           mean cross‑entropy loss.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_final_kernel(
    partials_ptr,    # float32 [NUM_PROGRAMS]
    out_ptr,         # float32 [1]
    R: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < NUM_PROGRAMS
    vals = tl.load(partials_ptr + offs, mask=mask, other=0.0)
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
    returns: float32 scalar = mean cross‑entropy loss
    """
    R, N = logits.shape
    assert targets.shape == (R,)
    assert logits.dtype == torch.float32
    assert targets.dtype == torch.int64

    # ---------- tuning constants ----------
    NUM_PROGRAMS = 1024          # grid‑stride count – divides R=8192
    BLOCK_SIZE_N = 2048          # divides N=32768 exactly

    # ---------- launch settings ----------
    NUM_WARPS_PERS = 8           # each program = 256 threads
    NUM_WARPS_FINAL = 32         # 1024‑thread reduction block

    # persistent kernel writes partial sums
    partials = torch.empty(NUM_PROGRAMS, dtype=torch.float32, device=logits.device)
    cross_entropy_persistent_kernel[(NUM_PROGRAMS,)](
        logits, targets, partials,
        R, N, NUM_PROGRAMS, BLOCK_SIZE_N,
        num_warps=NUM_WARPS_PERS,
    )

    # final reduction – single block
    out = torch.empty((), dtype=torch.float32, device=logits.device)
    BLOCK_SIZE_FINAL = NUM_PROGRAMS   # exactly covers all partials
    reduce_final_kernel[(1,)](
        partials, out,
        R, NUM_PROGRAMS, BLOCK_SIZE_FINAL,
        num_warps=NUM_WARPS_FINAL,
    )

    return out