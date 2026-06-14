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
    # each program handles exactly one row
    base = i * K

    # online softmax (single-pass) to compute max and logsumexp over the row
    m = tl.full([], float("-inf"), dtype=tl.float32)
    s = tl.full([], 0.0, dtype=tl.float32)

    for col_start in range(0, K, BLOCK_SIZE):
        # contiguous, vectorized load of a tile
        offs = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < K
        # use streaming cache hint to avoid L1 pollution for large logits tensor
        tile = tl.load(logits_ptr + base + offs, mask=mask, other=float("-inf"),
                       cache_modifier=".cg")
        tile_max = tl.max(tile, axis=0)
        m_new = tl.maximum(m, tile_max)
        # online softmax rescaling
        alpha = tl.exp(m - m_new)
        tile_exp = tl.exp(tile - m_new)
        s = s * alpha + tl.sum(tile_exp, axis=0)
        m = m_new

    # load the target logit value
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

    # accumulators
    loss_sum = torch.zeros(1, dtype=torch.float32, device=logits.device)
    count = torch.zeros(1, dtype=torch.int32, device=logits.device)
    output = torch.empty((), dtype=torch.float32, device=logits.device)

    # launch with a wide vectorized tile (BLOCK_SIZE multiple of 128B)
    # num_warps=8 gives 256 threads, processing 4096 elements per iteration
    BLOCK_SIZE = 4096
    grid = (M,)
    cross_entropy_kernel[grid](
        logits, targets, loss_sum, count, M, K, BLOCK_SIZE,
        num_warps=8,
    )

    # final reduction to mean loss
    final_mean_kernel[(1,)](loss_sum, count, output)

    return output