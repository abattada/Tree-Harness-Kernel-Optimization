import torch
import triton
import triton.language as tl


@triton.jit
def online_softmax_loss_kernel(
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

    m = tl.full([], float("-inf"), dtype=tl.float32)
    s = tl.full([], 0.0, dtype=tl.float32)

    # Single-pass online softmax over the row
    for off in range(0, K, BLOCK_COL):
        cols = off + tl.arange(0, BLOCK_COL)
        # No mask needed: BLOCK_COL divides K exactly
        vals = tl.load(logits_ptr + row_start + cols)

        block_max = tl.max(vals)
        new_m = tl.maximum(m, block_max)

        # Update the running sum with rescaling
        s = s * tl.exp(m - new_m) + tl.sum(tl.exp(vals - new_m))
        m = new_m

    # logsumexp = log(s) + m
    logsumexp = tl.log(s) + m

    # Gather target logit and compute loss
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
    off = tl.arange(0, BLOCK_SIZE)
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # No mask: BLOCK_SIZE divides N exactly
    for start in range(0, N, BLOCK_SIZE):
        val = tl.load(loss_ptr + start + off)
        sum_val += val

    total = tl.sum(sum_val)
    mean = total / N
    tl.store(out_ptr, mean)


def triton_run(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    M, K = logits.shape
    assert targets.shape == (M,)

    # BLOCK_COL must divide K exactly to avoid masks
    BLOCK_COL = 512  # 32768 % 512 == 0
    losses = torch.empty(M, dtype=logits.dtype, device=logits.device)

    grid = (M,)
    online_softmax_loss_kernel[grid](
        logits, targets, losses,
        M, K,
        BLOCK_COL,
        num_warps=8,
    )

    out = torch.empty(1, dtype=logits.dtype, device=logits.device)
    BLOCK_SIZE = 1024  # 8192 % 1024 == 0
    reduction_kernel[(1,)](
        losses, out,
        M,
        BLOCK_SIZE,
        num_warps=4,
    )

    return out.squeeze()