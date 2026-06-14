import torch
import triton
import triton.language as tl

@triton.jit
def cross_entropy_kernel(
    logits_ptr, targets_ptr, output_ptr,
    N: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    """One program per row: online softmax to compute per-row cross-entropy loss."""
    row_idx = tl.program_id(0)
    # target index for this row (int32 for fast arithmetic)
    target_idx = tl.cast(tl.load(targets_ptr + row_idx), tl.int32)

    max_val = -float('inf')
    sum_val = 0.0
    target_logit = 0.0
    row_start = row_idx * N

    # process the row in tiles
    for start in range(0, N, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(logits_ptr + row_start + offsets, mask=mask, other=-float('inf'))

        # tile-level max
        tile_max = tl.max(x, axis=0)  # out-of-bound are -inf → ignored
        new_max = tl.maximum(max_val, tile_max)

        # update sum with online formula
        exp_vals = tl.exp(x - new_max)
        sum_tile = tl.sum(tl.where(mask, exp_vals, 0.0))
        sum_val = sum_val * tl.exp(max_val - new_max) + sum_tile
        max_val = new_max

        # capture the exact logit corresponding to the target
        if target_idx >= start and target_idx < start + BLOCK_SIZE:
            target_logit = tl.load(logits_ptr + row_start + target_idx)

    # compute logsumexp and loss
    logsumexp = max_val + tl.log(sum_val)
    loss = -(target_logit - logsumexp)
    tl.store(output_ptr + row_idx, loss)


@triton.jit
def reduce_mean_kernel(input_ptr, output_ptr, N: tl.int32):
    """Single‑block reduction: sum all per‑row losses and divide by N."""
    pid = tl.program_id(0)
    if pid == 0:
        total = 0.0
        for i in range(N):
            total += tl.load(input_ptr + i)
        output_ptr[0] = total / N


def triton_run(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute mean cross‑entropy loss using softmax."""
    assert logits.ndim == 2 and targets.ndim == 1
    N, M = logits.shape
    assert targets.shape[0] == N

    BLOCK_SIZE = 2048  # tunable; divides 32768

    # per‑row losses
    per_row = torch.empty(N, dtype=torch.float32, device='cuda')

    # launch per‑row kernel
    grid_cross = (N,)
    cross_entropy_kernel[grid_cross](
        logits, targets, per_row, M,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=3,
    )

    # reduction to scalar
    out = torch.empty(1, dtype=torch.float32, device='cuda')
    grid_red = (1,)
    reduce_mean_kernel[grid_red](
        per_row, out, N,
        num_warps=1,
        num_stages=1,
    )

    return out