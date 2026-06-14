import torch
import triton
import triton.language as tl


@triton.jit
def row_online_softmax_loss_kernel(
    logits_ptr,
    targets_ptr,
    loss_ptr,
    M,
    K,
    BLOCK_COL: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return

    target = tl.load(targets_ptr + pid).to(tl.int32)
    row_start = pid * K

    # Online softmax: single pass for max and sum exp
    m_cur = tl.full([], -float('inf'), dtype=tl.float32)
    s_cur = tl.full([], 0.0, dtype=tl.float32)

    for off in range(0, K, BLOCK_COL):
        cols = off + tl.arange(0, BLOCK_COL)
        mask = cols < K
        block = tl.load(logits_ptr + row_start + cols, mask=mask, other=-float('inf'))

        block_max = tl.max(block)
        new_max = tl.maximum(m_cur, block_max)

        # Rescale running sum if max changed
        s_cur = s_cur * tl.exp(m_cur - new_max)

        # Accumulate exp of current block
        exp_block = tl.exp(block - new_max)
        s_cur = s_cur + tl.sum(tl.where(mask, exp_block, 0.0))

        m_cur = new_max

    logsumexp = tl.log(s_cur) + m_cur
    target_logit = tl.load(logits_ptr + row_start + target)
    loss = logsumexp - target_logit
    tl.store(loss_ptr + pid, loss)


@triton.jit
def reduction_kernel(
    loss_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    off = tl.arange(0, BLOCK_SIZE)
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for start in range(0, N, BLOCK_SIZE):
        idx = start + off
        mask = idx < N
        val = tl.load(loss_ptr + idx, mask=mask, other=0.0)
        sum_val += val

    total = tl.sum(sum_val)
    if pid == 0:
        mean = total / N
        tl.store(out_ptr, mean)


def triton_run(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    M, K = logits.shape
    assert targets.shape == (M,)

    # Allocate per-row losses
    losses = torch.empty(M, dtype=logits.dtype, device=logits.device)

    # Row-wise loss computation with online softmax (single pass over logits)
    BLOCK_COL = 256
    grid = (M,)
    row_online_softmax_loss_kernel[grid](
        logits, targets, losses,
        M, K,
        BLOCK_COL,
        num_warps=8,
    )

    # Final mean reduction over rows
    out = torch.empty(1, dtype=logits.dtype, device=logits.device)
    BLOCK_SIZE = 1024
    reduction_kernel[(1,)](
        losses, out,
        M,
        BLOCK_SIZE,
        num_warps=4,
    )

    return out.squeeze()