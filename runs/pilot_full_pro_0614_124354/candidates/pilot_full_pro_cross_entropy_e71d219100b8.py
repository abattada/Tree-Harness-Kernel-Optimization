import torch
import triton
import triton.language as tl


@triton.jit
def row_softmax_loss_kernel(
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

    # Load target index for this row
    target = tl.load(targets_ptr + pid).to(tl.int32)
    row_start = pid * K

    # ---------- first pass: row max ----------
    max_val = tl.full([], -float('inf'), dtype=tl.float32)
    for off in range(0, K, BLOCK_COL):
        cols = off + tl.arange(0, BLOCK_COL)
        mask = cols < K
        vals = tl.load(logits_ptr + row_start + cols, mask=mask, other=-float('inf'))
        block_max = tl.max(vals)
        max_val = tl.maximum(max_val, block_max)

    # ---------- second pass: sum exp(x - max) ----------
    sum_exp = tl.full([], 0.0, dtype=tl.float32)
    for off in range(0, K, BLOCK_COL):
        cols = off + tl.arange(0, BLOCK_COL)
        mask = cols < K
        vals = tl.load(logits_ptr + row_start + cols, mask=mask, other=0.0)
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(tl.where(mask, exp_vals, 0.0))

    logsumexp = tl.log(sum_exp) + max_val
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

    # Row-wise log-softmax and gather
    BLOCK_COL = 256
    grid = (M,)
    row_softmax_loss_kernel[grid](
        logits, targets, losses,
        M, K,
        BLOCK_COL,
        num_warps=8,
    )

    # Final mean reduction
    out = torch.empty(1, dtype=logits.dtype, device=logits.device)
    BLOCK_SIZE = 1024
    reduction_kernel[(1,)](
        losses, out,
        M,
        BLOCK_SIZE,
        num_warps=4,
    )

    return out.squeeze()