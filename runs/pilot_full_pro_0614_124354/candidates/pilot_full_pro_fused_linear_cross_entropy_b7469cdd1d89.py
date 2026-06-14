import torch
import triton
import triton.language as tl

@triton.jit
def flce_kernel(x_ptr, w_ptr, targets_ptr, loss_out_ptr, B, N, K,
                BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """
    Compute per-row cross-entropy loss of (x @ w.T) without materializing the full logits.

    Grid: (B,)  — one program per batch element.
    Each program:
      - Loops over output-class tiles (of size BLOCK_N) and inner dim tiles (BLOCK_K).
      - Uses online logsumexp (single pass) to compute logsumexp of logits.
      - Accumulates the target logit v_target.
      - Writes the final per-row loss to loss_out_ptr[row_idx].
    """
    row_idx = tl.program_id(0)
    if row_idx >= B:
        return

    target = tl.load(targets_ptr + row_idx)
    x_row = x_ptr + row_idx * K

    # online logsumexp state
    m = tl.full((1,), -float('inf'), dtype=tl.float32)
    s = tl.zeros((1,), dtype=tl.float32)
    v_target = tl.zeros((1,), dtype=tl.float32)

    for n_start in range(0, N, BLOCK_N):
        n_curr = min(BLOCK_N, N - n_start)

        # accumulate logits for this output-class tile
        logits = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            k_curr = min(BLOCK_K, K - k_start)

            # load x tile [BLOCK_K]
            x_offs = k_start + tl.arange(0, BLOCK_K)
            x_mask = x_offs < K
            x_tile = tl.load(x_row + x_offs, mask=x_mask, other=0.0)

            # load weight tile [BLOCK_N, BLOCK_K]
            n_offs = n_start + tl.arange(0, BLOCK_N)
            n_mask = n_offs < N
            w_ptrs = w_ptr + n_offs[:, None] * K + x_offs[None, :]
            w_tile = tl.load(w_ptrs, mask=n_mask[:, None] & x_mask[None, :], other=0.0)

            # partial dot products for this tile
            partial = tl.sum(w_tile * x_tile[None, :], axis=1)
            logits += partial

        # mask for valid output-class indices in this tile
        valid_mask = tl.arange(0, BLOCK_N) < n_curr

        # tile max and sumexp (with shift by max)
        m_tile = tl.max(tl.where(valid_mask, logits, -float('inf')))
        exp_tile = tl.exp(logits - m_tile)
        s_tile = tl.sum(tl.where(valid_mask, exp_tile, 0.0))

        # update online logsumexp
        m_new = tl.maximum(m, m_tile)
        s = s * tl.exp(m - m_new) + s_tile * tl.exp(m_tile - m_new)
        m = m_new

        # if target lies in this tile, accumulate its logit
        is_in_tile = (target >= n_start) & (target < n_start + n_curr)
        if is_in_tile:
            local_idx = target - n_start
            target_mask = tl.arange(0, BLOCK_N) == local_idx
            target_logit = tl.sum(tl.where(target_mask, logits, 0.0))
            v_target += target_logit

    # cross-entropy loss for this row: -v_target + logsumexp
    loss = -v_target + m + tl.log(s)
    tl.store(loss_out_ptr + row_idx, loss)


def triton_run(x: torch.Tensor, w: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Fused linear cross-entropy: mean( CE( x @ w.T, targets ) ) without materializing full logits.
    """
    B, K = x.shape
    N, K2 = w.shape
    assert K == K2 and targets.shape == (B,), "Invalid shapes"

    loss_per_row = torch.empty(B, dtype=torch.float32, device=x.device)

    # tuneable parameters (obvious knobs for later tuning)
    BLOCK_N = 128
    BLOCK_K = 128
    num_warps = 4
    num_stages = 3

    grid = (B,)
    flce_kernel[grid](
        x, w, targets, loss_per_row,
        B, N, K,
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )

    return loss_per_row.mean()