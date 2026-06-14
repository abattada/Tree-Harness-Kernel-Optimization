import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 1: per‑row online softmax + nll loss, one program per row.
# N == 32768 is exactly divisible by BLOCK_SIZE -> mask is removed.
# ---------------------------------------------------------------------------
@triton.jit
def cross_entropy_row_kernel(
    logits_ptr,      # f32 [R, N]
    targets_ptr,     # i64 [R]
    loss_row_ptr,    # f32 [R]   per‑row loss, to be reduced later
    N: tl.constexpr,           # number of classes
    BLOCK_SIZE: tl.constexpr,  # block along N dimension
):
    pid = tl.program_id(0)
    target = tl.load(targets_ptr + pid)

    # Online softmax: single pass for logsumexp
    m_old = tl.full([], float('-inf'), dtype=tl.float32)
    d_old = tl.full([], 0.0, dtype=tl.float32)

    row_base = pid * N
    for start in range(0, N, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        # N % BLOCK_SIZE == 0, so no mask needed
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
    tl.store(loss_row_ptr + pid, loss, eviction_policy='evict_last')


# ---------------------------------------------------------------------------
# First reduction stage: sum per‑row losses in fixed-size groups.
# R (8192) is exactly divisible by BLOCK_SIZE (1024) -> no mask.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_sum_stage1_kernel(
    inp_ptr,          # f32 [R]
    out_partial_ptr,  # f32 [num_partials]
    BLOCK_SIZE: tl.constexpr,  # rows per program (1024)
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    vals = tl.load(inp_ptr + offs, eviction_policy='evict_first')
    partial = tl.sum(vals, axis=0)
    tl.store(out_partial_ptr + pid, partial, eviction_policy='evict_last')


# ---------------------------------------------------------------------------
# Second reduction stage: sum all partials and divide by R.
# num_partials is known (8) and we set BLOCK_SIZE equal to it → no mask.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_mean_stage2_kernel(
    inp_ptr,          # f32 [num_partials]
    out_scalar_ptr,   # f32 [1]
    R: tl.constexpr,           # total rows (for division)
    BLOCK_SIZE: tl.constexpr,  # == num_partials
):
    offs = tl.arange(0, BLOCK_SIZE)
    vals = tl.load(inp_ptr + offs, eviction_policy='evict_first')
    total = tl.sum(vals, axis=0)
    mean = total / R
    tl.store(out_scalar_ptr, mean)


# ---------------------------------------------------------------------------
# Top‑level driver
# ---------------------------------------------------------------------------
def triton_run(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    R, N = logits.shape
    assert logits.dtype == torch.float32
    assert targets.dtype == torch.int64
    assert targets.shape[0] == R

    # ----- Row kernel -----
    loss_row = torch.empty(R, device='cuda', dtype=torch.float32)
    BLOCK_SIZE = 1024  # divides N=32768 exactly
    grid = (R,)
    cross_entropy_row_kernel[grid](
        logits, targets, loss_row,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
    )

    # ----- First reduction stage -----
    BLOCK_SIZE_STAGE1 = 1024  # divides R=8192 exactly → grid = 8
    num_partials = R // BLOCK_SIZE_STAGE1
    partials = torch.empty(num_partials, device='cuda', dtype=torch.float32)

    reduce_sum_stage1_kernel[(num_partials,)](
        loss_row, partials,
        BLOCK_SIZE=BLOCK_SIZE_STAGE1,
        num_warps=4,
        num_stages=2,
    )

    # ----- Second reduction stage -----
    out = torch.empty(1, device='cuda', dtype=torch.float32)
    BLOCK_SIZE_STAGE2 = num_partials  # 8
    reduce_mean_stage2_kernel[(1,)](
        partials, out,
        R=R,
        BLOCK_SIZE=BLOCK_SIZE_STAGE2,
        num_warps=4,
        num_stages=2,
    )

    return out.squeeze()