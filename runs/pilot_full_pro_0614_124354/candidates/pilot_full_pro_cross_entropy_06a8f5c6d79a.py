import torch
import triton
import triton.language as tl


@triton.jit
def cross_entropy_row_kernel(logits_ptr, targets_ptr, loss_ptr, n_cols,
                             BLOCK: tl.constexpr):
    row = tl.program_id(0)
    target = tl.load(targets_ptr + row).to(tl.int32)
    base = logits_ptr + row * n_cols

    # Pass 1: find row maximum
    max_val = tl.full((1,), -float('inf'), dtype=tl.float32)
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(base + cols, mask=mask, other=-float('inf'))
        block_max = tl.max(x)
        max_val = tl.maximum(max_val, block_max)

    # Pass 2: sum of exp(x - max)
    sum_exp = tl.zeros((1,), dtype=tl.float32)
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(base + cols, mask=mask, other=-float('inf'))
        x_shifted = x - max_val
        exp_x = tl.exp(x_shifted)
        block_sum = tl.sum(exp_x)
        sum_exp += block_sum

    lse = max_val + tl.log(sum_exp)
    target_logit = tl.load(base + target, mask=True, other=0.0)
    loss = lse - target_logit
    tl.store(loss_ptr + row, loss)


@triton.jit
def reduce_sum_kernel(in_ptr, out_ptr, N, BLOCK: tl.constexpr):
    total = tl.zeros((1,), dtype=tl.float32)
    for start in range(0, N, BLOCK):
        cols = tl.arange(0, BLOCK) + start
        mask = cols < N
        vals = tl.load(in_ptr + cols, mask=mask, other=0.0)
        total += tl.sum(vals)
    tl.store(out_ptr, total)


def triton_run(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits: f32[8192, 32768]
    targets: i64[8192]
    returns: f32[]  (scalar mean cross-entropy loss)
    """
    N, C = logits.shape
    device = logits.device

    # Allocate per-row loss
    per_row_loss = torch.empty(N, dtype=torch.float32, device=device)

    # Launch per-row cross-entropy kernel (one block per row)
    BLOCK = 1024
    grid = (N,)
    cross_entropy_row_kernel[grid](logits, targets, per_row_loss, C, BLOCK,
                                   num_warps=8, num_stages=2)

    # Sum the per-row losses in a single block
    BLOCK_RED = 2048
    out_sum = torch.empty(1, dtype=torch.float32, device=device)
    reduce_sum_kernel[(1,)](per_row_loss, out_sum, N, BLOCK_RED,
                            num_warps=4, num_stages=2)

    mean_loss = out_sum[0] / N
    return mean_loss