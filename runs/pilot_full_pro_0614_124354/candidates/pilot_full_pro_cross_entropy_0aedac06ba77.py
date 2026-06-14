import torch
import triton
import triton.language as tl


@triton.jit
def cross_entropy_kernel(
    logits_ptr,
    targets_ptr,
    loss_sum_ptr,
    count_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * K  # row start is always 128B-aligned (K % 32 = 0)

    # online softmax – single-pass max + logsumexp
    m = tl.full((), float("-inf"), dtype=tl.float32)
    s = tl.full((), 0.0, dtype=tl.float32)

    # tile loop: every tile load is 128B-aligned and contiguous
    for col_start in range(0, K, BLOCK_SIZE):
        offs = col_start + tl.arange(0, BLOCK_SIZE)
        # K is a perfect multiple of BLOCK_SIZE, so we can omit the mask;
        # we keep it for safety if the shape ever changes (does not affect perf).
        mask = offs < K
        tile = tl.load(
            logits_ptr + row_start + offs,
            mask=mask,
            other=float("-inf"),
            cache_modifier=".cg",          # streaming hint for this huge tensor
        )
        tile_max = tl.max(tile)
        m_new = tl.maximum(m, tile_max)
        scale = tl.exp(m - m_new)
        s = s * scale + tl.sum(tl.exp(tile - m_new))
        m = m_new

    target_idx = tl.load(targets_ptr + pid).to(tl.int32)
    target_val = tl.load(logits_ptr + row_start + target_idx, cache_modifier=".cg")
    # cross-entropy for this row: -(target - logsumexp)
    loss = m + tl.log(s) - target_val

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

    loss_sum = torch.zeros(1, dtype=torch.float32, device=logits.device)
    count = torch.zeros(1, dtype=torch.int32, device=logits.device)
    output = torch.empty((), dtype=torch.float32, device=logits.device)

    # 2048 elements with 2 warps → 32 elements/thread = 128 B loads
    BLOCK_SIZE = 2048
    num_warps = 2
    cross_entropy_kernel[(M,)](
        logits, targets, loss_sum, count,
        M, K, BLOCK_SIZE,
        num_warps=num_warps,
    )
    final_mean_kernel[(1,)](loss_sum, count, output)

    return output