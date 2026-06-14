import torch
import triton
import triton.language as tl


@triton.jit
def cross_entropy_row_kernel(
    logits_ptr,      # f32 [R, N]
    targets_ptr,     # i64 [R]
    loss_row_ptr,    # f32 [R]
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    target = tl.load(targets_ptr + pid)

    m_i = tl.full([], float('-inf'), dtype=tl.float32)
    d_i = tl.full([], 0.0, dtype=tl.float32)

    row_base = pid * N
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(logits_ptr + row_base + offs, mask=mask, other=float('-inf'))

        m_curr = tl.max(x, axis=0)
        m_new = tl.maximum(m_i, m_curr)
        exp_x = tl.exp(x - m_new)
        sum_exp = tl.sum(exp_x, axis=0)

        d_i = d_i * tl.exp(m_i - m_new) + sum_exp
        m_i = m_new

    logsumexp = m_i + tl.log(d_i)
    target_logit = tl.load(logits_ptr + row_base + target)
    loss = logsumexp - target_logit
    tl.store(loss_row_ptr + pid, loss)


@triton.jit
def reduce_sum_stage1_kernel(
    inp_ptr,           # f32 [R]
    out_partial_ptr,   # f32 [num_partials]
    R: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < R
    vals = tl.load(inp_ptr + offs, mask=mask, other=0.0)
    partial = tl.sum(vals, axis=0)
    tl.store(out_partial_ptr + pid, partial)


@triton.jit
def reduce_mean_stage2_kernel(
    inp_ptr,           # f32 [num_partials]
    out_scalar_ptr,    # f32 [1]
    R: tl.constexpr,
    num_partials: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < num_partials
    vals = tl.load(inp_ptr + offs, mask=mask, other=0.0)
    total = tl.sum(vals, axis=0)
    mean = total / R
    tl.store(out_scalar_ptr, mean)


def triton_run(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    R, N = logits.shape
    assert logits.dtype == torch.float32
    assert targets.dtype == torch.int64
    assert targets.shape == (R,)

    # Output is a scalar (mean cross-entropy)
    out = torch.empty(1, dtype=torch.float32, device=logits.device)

    # Stage 0: per-row online softmax + NLL loss
    BLOCK_N = 2048  # N=32768 is a multiple of 2048
    loss_row = torch.empty(R, dtype=torch.float32, device=logits.device)
    cross_entropy_row_kernel[(R,)](
        logits, targets, loss_row,
        N, BLOCK_N,
        num_warps=8, num_stages=3,
    )

    # Stage 1: partial sum of per-row losses
    BLOCK1 = 128
    num_partials = (R + BLOCK1 - 1) // BLOCK1
    partials = torch.empty(num_partials, dtype=torch.float32, device=logits.device)
    reduce_sum_stage1_kernel[(num_partials,)](
        loss_row, partials,
        R, BLOCK1,
        num_warps=4,
    )

    # Stage 2: final sum and mean
    BLOCK2 = 1
    while BLOCK2 < num_partials:
        BLOCK2 *= 2
    reduce_mean_stage2_kernel[(1,)](
        partials, out,
        R, num_partials, BLOCK2,
        num_warps=1,
    )

    return out.reshape([])