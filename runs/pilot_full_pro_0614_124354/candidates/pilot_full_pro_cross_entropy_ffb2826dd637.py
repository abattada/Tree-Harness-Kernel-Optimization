import torch
import triton
import triton.language as tl


@triton.jit
def cross_entropy_kernel(
    logits_ptr,
    targets_ptr,
    loss_sum_ptr,
    count_ptr,
    M,
    K,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid
    # one program per row, grid size = M
    base = i * K

    # online softmax (single-pass) to compute max and logsumexp
    m = tl.full([], float("-inf"), dtype=tl.float32)
    s = tl.full([], 0.0, dtype=tl.float32)

    for col_start in range(0, K, BLOCK_SIZE):
        offs = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < K
        tile = tl.load(logits_ptr + base + offs, mask=mask, other=float("-inf"))
        tile_max = tl.max(tile, axis=0)
        m_new = tl.maximum(m, tile_max)
        scale = tl.exp(m - m_new)
        s = s * scale + tl.sum(tl.exp(tile - m_new), axis=0)
        m = m_new

    # load the target logit
    target_idx = tl.load(targets_ptr + i).to(tl.int32)
    target_val = tl.load(logits_ptr + base + target_idx)

    # cross-entropy loss for this row
    loss = -(target_val - m - tl.log(s))

    # accumulate into global mean via atomics
    tl.atomic_add(loss_sum_ptr, loss)
    tl.atomic_add(count_ptr, 1)


@triton.jit
def final_mean_kernel(loss_sum_ptr, count_ptr, output_ptr):
    loss_sum = tl.load(loss_sum_ptr)
    count = tl.load(count_ptr).to(tl.float32)
    tl.store(output_ptr, loss_sum / count)


def triton_run(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    M, K = logits.shape
    assert targets.shape == (M,)

    # accumulators (initialised to zero)
    loss_sum = torch.zeros(1, dtype=torch.float32, device=logits.device)
    count = torch.zeros(1, dtype=torch.int32, device=logits.device)

    # output scalar
    output = torch.empty((), dtype=torch.float32, device=logits.device)

    # launch row-loss kernel over all rows
    BLOCK_SIZE = 1024
    grid = (M,)
    cross_entropy_kernel[grid](
        logits, targets, loss_sum, count, M, K, BLOCK_SIZE, num_warps=4
    )

    # final division to obtain mean loss
    final_mean_kernel[(1,)](loss_sum, count, output)

    return output