import torch
import triton
import triton.language as tl

@triton.jit
def _fused_linear_ce_kernel(
    x_ptr, w_ptr, target_ptr, loss_per_row_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    row_offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = row_offs < M

    # Load target indices for these rows
    targets = tl.load(target_ptr + row_offs, mask=mask_m, other=0)

    # Online softmax state (use large finite negative instead of -inf to avoid NaN)
    max_vals = tl.full([BLOCK_M], -1e30, dtype=tl.float32)
    sumexps = tl.zeros([BLOCK_M], dtype=tl.float32)
    target_scores = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Loop over output classes in tiles
    for n_start in range(0, N, BLOCK_N):
        n_offs = n_start + tl.arange(0, BLOCK_N)
        mask_n = n_offs < N

        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        # Inner reduction over K
        for k_start in range(0, K, BLOCK_K):
            k_offs = k_start + tl.arange(0, BLOCK_K)
            mask_k = k_offs < K

            # Load x tile [BLOCK_M, BLOCK_K]
            x_ptrs = x_ptr + row_offs[:, None] * stride_xm + k_offs[None, :] * stride_xk
            x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

            # Load w tile [BLOCK_K, BLOCK_N]  (K fast index -> coalesced)
            w_ptrs = w_ptr + k_offs[:, None] * stride_wk + n_offs[None, :] * stride_wn
            w_tile = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

            # Accumulate dot product
            acc += tl.dot(x_tile, w_tile)

        # Rows outside M should not influence the softmax statistics
        acc = tl.where(mask_m[:, None], acc, -float('inf'))

        # Online softmax update for this N‑block
        row_max_curr = tl.max(acc, axis=1)
        new_max = tl.maximum(max_vals, row_max_curr)
        old_max = max_vals
        exp_adjust = tl.exp(old_max - new_max)
        sumexps = sumexps * exp_adjust + tl.sum(tl.exp(acc - new_max[:, None]), axis=1)
        max_vals = new_max

        # If the target class is in the current tile, accumulate its score
        in_block = (targets >= n_start) & (targets < n_start + BLOCK_N) & mask_m
        target_col_offs = targets - n_start
        mask_target = (tl.arange(0, BLOCK_N)[None, :] == target_col_offs[:, None]) & in_block[:, None]
        target_scores += tl.sum(acc * mask_target.to(tl.float32), axis=1)

    # Final log-sum-exp and per‑row cross entropy
    logsumexp = max_vals + tl.log(sumexps)
    loss = - (target_scores - logsumexp)

    # Output per‑row losses (the mean over rows is taken outside the kernel)
    tl.store(loss_per_row_ptr + row_offs, loss, mask=mask_m)


def triton_run(x: torch.Tensor, w: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    M, K = x.shape
    N, K_ = w.shape
    assert K == K_, "Inner dimension mismatch"

    loss_per_row = torch.empty(M, dtype=torch.float32, device=x.device)

    # Tiling tuned for RTX 5090 (occupancy vs. register pressure)
    BLOCK_M = 32
    BLOCK_N = 128
    BLOCK_K = 64
    num_warps = 8
    num_stages = 3

    grid = (triton.cdiv(M, BLOCK_M),)

    _fused_linear_ce_kernel[grid](
        x, w, targets, loss_per_row,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    # Trivial glue: mean reduction of the already computed per‑row losses
    return loss_per_row.mean()