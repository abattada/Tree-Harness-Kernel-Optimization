import torch
import triton
import triton.language as tl

# Tuning knobs: block size for the C dimension (row kernel) and reduction block.
# These are constants known at compile time.
BLOCK_C = 1024
BLOCK_RED = 256


@triton.jit
def cross_entropy_row_kernel(logits_ptr, targets_ptr, losses_ptr,
                             C: tl.constexpr, BLOCK_C: tl.constexpr):
    """
    Computes the cross-entropy loss for a single row.
    Uses an online stable softmax algorithm to avoid overflow and minimise passes.
    """
    row_idx = tl.program_id(0)
    base = logits_ptr + row_idx * C

    # Online stable softmax state
    M = -float('inf')
    S = 0.0

    for start in range(0, C, BLOCK_C):
        off = tl.arange(0, BLOCK_C)
        idx = start + off
        mask = idx < C
        logit_block = tl.load(base + idx, mask=mask, other=-float('inf'))

        M_curr = tl.max(logit_block)
        if M_curr > -float('inf'):            # only if the block is not fully masked
            M_new = tl.maximum(M, M_curr)
            correction = tl.exp(M - M_new)    # = exp(old_M - new_M), 0 when M = -inf
            S = S * correction + tl.sum(tl.exp(logit_block - M_new))
            M = M_new

    logsumexp = M + tl.log(S)

    # Read the target class logit (int64 index cast to int32 for Triton pointer arithmetic)
    target = tl.load(targets_ptr + row_idx).to(tl.int32)
    target_logit = tl.load(logits_ptr + row_idx * C + target)

    loss = -(target_logit - logsumexp)
    tl.store(losses_ptr + row_idx, loss)


@triton.jit
def reduce_sum_kernel(losses_ptr, output_ptr, N: tl.constexpr, BLOCK_RED: tl.constexpr):
    """
    Parallel sum reduction: each block reduces a chunk of 'losses' and atomically adds
    its partial sum to the scalar output.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_RED
    off = tl.arange(0, BLOCK_RED)
    idx = block_start + off
    mask = idx < N
    vals = tl.load(losses_ptr + idx, mask=mask, other=0.0)
    partial_sum = tl.sum(vals)
    tl.atomic_add(output_ptr, partial_sum)


def triton_run(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits: f32[8192, 32768]
    targets: i64[8192]
    returns: f32[]  (mean cross-entropy loss)
    """
    N, C = logits.shape
    assert N == 8192 and C == 32768, "Unexpected shapes"

    losses = torch.empty(N, dtype=torch.float32, device=logits.device)

    # Step 1: per-row cross-entropy (produces N loss scalars)
    grid = (N,)
    cross_entropy_row_kernel[grid](
        logits, targets, losses,
        C=C, BLOCK_C=BLOCK_C,
        num_warps=4, num_stages=4
    )

    # Step 2: sum the per-row losses (reduce into a single-element tensor)
    sum_val = torch.zeros(1, dtype=torch.float32, device=logits.device)
    grid_reduce = (triton.cdiv(N, BLOCK_RED),)
    reduce_sum_kernel[grid_reduce](
        losses, sum_val,
        N=N, BLOCK_RED=BLOCK_RED,
        num_warps=4
    )

    # Step 3: final division to form the mean (simple torch operation)
    return sum_val / N