import torch
import triton
import triton.language as tl


@triton.jit
def _flce_kernel(
    x_ptr,
    w_ptr,
    targets_ptr,
    loss_ptr,
    B,
    V,
    D,
    BLOCK_M: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    row_mask = rows < B

    # Load target indices for this block of rows
    tidx = tl.load(targets_ptr + rows, mask=row_mask, other=0)

    # Per-row online softmax state
    m = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l = tl.zeros([BLOCK_M], dtype=tl.float32)
    t_logit = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Outer loop over vocabulary (V) tiles
    for v_start in range(0, V, BLOCK_V):
        v_offs = v_start + tl.arange(0, BLOCK_V)
        v_mask = v_offs < V

        # Accumulate the dot-product for this (row chunk, vocab chunk)
        acc = tl.zeros([BLOCK_M, BLOCK_V], dtype=tl.float32)

        for d_start in range(0, D, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offs < D

            # x tile  (BLOCK_M, BLOCK_D)
            x_ptrs = x_ptr + rows[:, None] * D + d_offs[None, :]
            x_tile = tl.load(
                x_ptrs,
                mask=row_mask[:, None] & d_mask[None, :],
                other=0.0,
            )

            # w tile  (BLOCK_V, BLOCK_D)
            w_ptrs = w_ptr + v_offs[:, None] * D + d_offs[None, :]
            w_tile = tl.load(
                w_ptrs,
                mask=v_mask[:, None] & d_mask[None, :],
                other=0.0,
            )

            # Accumulate over the reduction dimension
            acc += tl.dot(x_tile, tl.trans(w_tile))

        # Full logits for this vocab tile (rows x current V chunk)
        logits = tl.where(v_mask[None, :], acc, float('-inf'))

        # Online softmax update
        tile_max = tl.max(logits, axis=1)
        new_max = tl.maximum(m, tile_max)

        exp_tile = tl.exp(logits - new_max[:, None])
        l_update = tl.sum(exp_tile, axis=1)
        l = l * tl.exp(m - new_max) + l_update
        m = new_max

        # If a target index falls inside the current V tile, extract its logit
        in_tile = (tidx >= v_start) & (tidx < v_start + BLOCK_V)
        for i in tl.static_range(BLOCK_M):
            if in_tile[i]:
                t_logit[i] = logits[i, tidx[i] - v_start]

    # Per-row cross-entropy loss: -(target_logit - logsumexp)
    loss_row = -(t_logit - m) + tl.log(l)
    # Mask out padding rows beyond B
    loss_row = tl.where(row_mask, loss_row, 0.0)

    # Sum across the rows handled by this block and atomically reduce
    block_loss = tl.sum(loss_row)
    if tl.thread_id() == 0:
        tl.atomic_add(loss_ptr, block_loss)


def triton_run(
    x: torch.Tensor, w: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """return mean cross-entropy of (x @ w.T) without materializing the full logits."""
    B, D = x.shape
    V = w.shape[0]
    # Hyperparameters (seeded for good occupancy on sm_120)
    BLOCK_M: tl.constexpr = 64
    BLOCK_V: tl.constexpr = 128
    BLOCK_D: tl.constexpr = 64
    num_warps = 8
    num_stages = 4

    loss_sum = torch.zeros(1, dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(B, BLOCK_M),)

    _flce_kernel[grid](
        x,
        w,
        targets,
        loss_sum,
        B,
        V,
        D,
        BLOCK_M=BLOCK_M,
        BLOCK_V=BLOCK_V,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return loss_sum / B