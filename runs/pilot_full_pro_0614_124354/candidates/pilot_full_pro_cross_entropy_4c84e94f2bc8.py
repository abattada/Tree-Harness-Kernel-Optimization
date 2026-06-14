import torch
import triton
import triton.language as tl


@triton.jit
def cross_entropy_kernel(
    logits_ptr,
    targets_ptr,
    loss_sum_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROG
    row_end = tl.minimum(row_start + ROWS_PER_PROG, M)

    loss_acc = tl.full([], 0.0, dtype=tl.float32)

    for i in range(row_start, row_end):
        base = i * K

        m = tl.full([], float("-inf"), dtype=tl.float32)
        s = tl.full([], 0.0, dtype=tl.float32)

        for col_start in range(0, K, BLOCK_SIZE):
            offs = col_start + tl.arange(0, BLOCK_SIZE)
            mask = offs < K
            tile = tl.load(logits_ptr + base + offs, mask=mask,
                           other=float("-inf"), eviction_policy="evict_first")
            tile_max = tl.max(tile, axis=0)
            m_new = tl.maximum(m, tile_max)
            scale = tl.exp(m - m_new)
            s = s * scale + tl.sum(tl.exp(tile - m_new), axis=0)
            m = m_new

        target_idx = tl.load(targets_ptr + i).to(tl.int32)
        target_val = tl.load(logits_ptr + base + target_idx)

        loss = -(target_val - m - tl.log(s))
        loss_acc += loss

    tl.atomic_add(loss_sum_ptr, loss_acc)


@triton.jit
def final_mean_kernel(loss_sum_ptr, output_ptr, M):
    loss_sum = tl.load(loss_sum_ptr)
    mean = loss_sum / tl.cast(M, tl.float32)
    tl.store(output_ptr, mean)


def triton_run(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    M, K = logits.shape
    assert targets.shape == (M,), "targets must be 1‑d of length M"

    BLOCK_SIZE = 1024
    ROWS_PER_PROG = 8
    grid = ((M + ROWS_PER_PROG - 1) // ROWS_PER_PROG,)

    loss_sum = torch.zeros(1, dtype=torch.float32, device=logits.device)
    output = torch.empty((), dtype=torch.float32, device=logits.device)

    cross_entropy_kernel[grid](
        logits, targets, loss_sum,
        M=M, K=K,
        BLOCK_SIZE=BLOCK_SIZE,
        ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=8,
        num_stages=3,
    )

    final_mean_kernel[(1,)](loss_sum, output, M)

    return output