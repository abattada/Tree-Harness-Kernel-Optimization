import torch
import triton
import triton.language as tl


@triton.jit
def cross_entropy_multirow_kernel(
    logits_ptr,      # f32 [R, N]
    targets_ptr,     # i64 [R]
    partials_ptr,    # f32 [GRID_SIZE]   one partial sum per program
    N: tl.constexpr,
    R: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GRID_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_sum = tl.zeros([], dtype=tl.float32)

    # Each program processes rows with stride GRID_SIZE.
    # R is a multiple of GRID_SIZE so no row-boundary mask is needed.
    for r in range(pid, R, GRID_SIZE):
        target = tl.load(targets_ptr + r)

        # Online softmax for this row
        m_curr = tl.full([], float('-inf'), dtype=tl.float32)
        d_curr = tl.full([], 0.0, dtype=tl.float32)
        row_base = r * N

        for start in range(0, N, BLOCK_SIZE_N):
            offs = start + tl.arange(0, BLOCK_SIZE_N)
            x = tl.load(logits_ptr + row_base + offs,
                        eviction_policy='evict_first')
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
        block_sum += loss

    tl.store(partials_ptr + pid, block_sum)


@triton.jit
def reduce_sum_mean_kernel(
    partials_ptr,      # f32 [num_partials]
    out_ptr,           # f32 []
    R: tl.constexpr,
    NUM_PARTIALS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    vals = tl.load(partials_ptr + offs,
                   mask=offs < NUM_PARTIALS,
                   other=0.0)
    total = tl.sum(vals, axis=0)
    mean = total / R
    tl.store(out_ptr, mean)


def triton_run(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits : f32 [8192, 32768]
    targets: i64 [8192]
    returns: f32 scalar = mean cross-entropy loss
    """
    R, N = logits.shape

    # Tuning parameters – all divide their respective dimension exactly
    BLOCK_SIZE_N = 2048          # splits N into 16 chunks
    GRID_SIZE    = 512            # 512 programs, each processes 16 rows

    partials = torch.empty(GRID_SIZE, dtype=torch.float32, device=logits.device)

    cross_entropy_multirow_kernel[(GRID_SIZE,)](
        logits, targets, partials,
        N=N, R=R,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        GRID_SIZE=GRID_SIZE,
        num_warps=8,              # 256 threads per block
    )

    out = torch.empty((), dtype=torch.float32, device=logits.device)

    reduce_sum_mean_kernel[(1,)](
        partials, out,
        R=R,
        NUM_PARTIALS=GRID_SIZE,
        BLOCK_SIZE=GRID_SIZE,
        num_warps=16,             # 512 threads, exactly one per partial
    )

    return out