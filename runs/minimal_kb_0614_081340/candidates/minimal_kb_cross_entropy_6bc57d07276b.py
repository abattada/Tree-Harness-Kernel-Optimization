import torch
import triton
import triton.language as tl
import math

@triton.jit
def cross_entropy_kernel(
    logits_ptr, targets_ptr, losses_ptr,
    B, C,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    # row start pointer
    row_logits = logits_ptr + pid * C
    target = tl.load(targets_ptr + pid)

    # online logsumexp
    running_max = tl.full([1], -float('inf'), dtype=tl.float32)
    running_sum = tl.full([1], 0.0, dtype=tl.float32)

    # loop over columns in blocks
    offs_c = tl.arange(0, BLOCK_C)
    for start in range(0, C, BLOCK_C):
        mask = offs_c < (C - start)
        x = tl.load(row_logits + start + offs_c, mask=mask, other=-float('inf'))
        block_max = tl.max(x, axis=0)
        block_sum = tl.sum(tl.exp(x - block_max), axis=0)

        # merge with running stats
        is_greater = block_max > running_max
        # update sum: if new max is larger, scale old sum; else scale new sum
        running_sum = tl.where(is_greater,
                               running_sum * tl.exp(running_max - block_max) + block_sum,
                               running_sum + block_sum * tl.exp(block_max - running_max))
        running_max = tl.maximum(running_max, block_max)

    logsumexp = running_max + tl.log(running_sum)

    # load target logit (single element)
    target_logit = tl.load(row_logits + target)
    loss = logsumexp - target_logit

    tl.store(losses_ptr + pid, loss)

@triton.jit
def reduce_sum_kernel(
    input_ptr, output_ptr,
    N,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    # grid-stride loop over input
    acc = tl.full([1], 0.0, dtype=tl.float32)
    offs = tl.arange(0, BLOCK_N)
    for start in range(0, N, BLOCK_N):
        mask = offs < (N - start)
        val = tl.load(input_ptr + start + offs, mask=mask, other=0.0)
        acc += tl.sum(val, axis=0)
    # single warp reduction to get final sum
    acc_sum = tl.sum(acc, axis=0)
    tl.store(output_ptr, acc_sum)

def triton_run(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    B, C = logits.shape
    # allocate per-row loss buffer
    batch_losses = torch.empty(B, dtype=torch.float32, device=logits.device)

    # launch first kernel: one program per row
    BLOCK_C = 2048  # tune: 2048 works for 32768 columns (16 iterations)
    grid = (B,)
    cross_entropy_kernel[grid](
        logits, targets, batch_losses,
        B, C,
        BLOCK_C,
        num_warps=4,
    )

    # second kernel: reduce to scalar and divide by batch size
    output = torch.empty(1, dtype=torch.float32, device=logits.device)
    BLOCK_N = 8192  # batch size is 8192, one block is enough
    reduce_sum_kernel[(1,)](
        batch_losses, output,
        B,
        BLOCK_N,
        num_warps=4,
    )
    # mean loss
    output /= B
    return output[0]