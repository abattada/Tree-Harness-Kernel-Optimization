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
    ROWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    # each program handles up to ROWS_PER_PROG rows starting at row_start
    row_start = pid * ROWS_PER_PROG

    # local accumulation for this program
    pg_loss_sum = 0.0
    pg_count = 0

    for r in range(ROWS_PER_PROG):
        row = row_start + r
        if row >= M:
            break

        base = row * K

        # online softmax for this row
        m_val = tl.full([], float("-inf"), dtype=tl.float32)
        s_val = tl.full([], 0.0, dtype=tl.float32)

        for col_start in range(0, K, BLOCK_SIZE):
            offs = col_start + tl.arange(0, BLOCK_SIZE)
            mask = offs < K
            tile = tl.load(logits_ptr + base + offs, mask=mask, other=float("-inf"))
            tile_max = tl.max(tile, axis=0)
            m_new = tl.maximum(m_val, tile_max)
            scale = tl.exp(m_val - m_new)
            s_val = s_val * scale + tl.sum(tl.exp(tile - m_new), axis=0)
            m_val = m_new

        # load target index and its logit
        target_idx = tl.load(targets_ptr + row).to(tl.int32)
        target_val = tl.load(logits_ptr + base + target_idx)

        # cross-entropy contribution
        loss = -(target_val - m_val - tl.log(s_val))
        pg_loss_sum += loss
        pg_count += 1

    # atomically update global accumulators
    if pg_count > 0:
        tl.atomic_add(loss_sum_ptr, pg_loss_sum)
        tl.atomic_add(count_ptr, pg_count)


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

    BLOCK_SIZE = 2048
    ROWS_PER_PROG = 8
    num_warps = 8

    grid = (triton.cdiv(M, ROWS_PER_PROG),)
    cross_entropy_kernel[grid](
        logits, targets, loss_sum, count, M, K,
        BLOCK_SIZE=BLOCK_SIZE, ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=num_warps,
    )

    final_mean_kernel[(1,)](loss_sum, count, output)

    return output