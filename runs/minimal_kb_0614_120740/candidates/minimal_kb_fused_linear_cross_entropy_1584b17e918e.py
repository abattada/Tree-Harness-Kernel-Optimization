import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 1: fused linear layer + cross-entropy (online softmax) per tile of rows.
# Each program handles BLOCK_M rows and processes all classes in BLOCK_N blocks.
# For each class block, it computes logits via a K-loop tile-based matrix-vector
# multiplication (dot product of input rows with weight rows), then updates
# per-row online softmax statistics (max and sumexp). The logit for the target
# class is captured when its block is processed. After all class blocks, per-row
# loss = logsumexp - target_logit is written to a global buffer.
# ---------------------------------------------------------------------------
@triton.jit
def fused_linear_ce_kernel(
    x_ptr,               # f32 [R, K]
    w_ptr,               # f32 [N, K]
    targets_ptr,         # i64 [R]
    loss_per_row_ptr,    # f32 [R]   output: per-row losses
    R: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    offsets_m = row_start + tl.arange(0, BLOCK_M)
    mask_m = offsets_m < R

    # load targets for these rows
    targets = tl.load(targets_ptr + offsets_m, mask=mask_m, other=0)

    # per-row online softmax state
    m = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    d = tl.zeros([BLOCK_M], dtype=tl.float32)
    logit_target = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Loop over class blocks
    for class_start in range(0, N, BLOCK_N):
        offsets_n = class_start + tl.arange(0, BLOCK_N)
        mask_n = offsets_n < N

        # accumulate logits for this class block
        logits_tile = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        # K-loop over hidden dimension
        for k_start in range(0, K, BLOCK_K):
            offsets_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offsets_k < K

            # load x tile: (BLOCK_M, BLOCK_K)
            x_tile = tl.load(
                x_ptr + offsets_m[:, None] * K + offsets_k[None, :],
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
                eviction_policy='evict_first',
            )

            # load w tile: (BLOCK_N, BLOCK_K)
            w_tile = tl.load(
                w_ptr + offsets_n[:, None] * K + offsets_k[None, :],
                mask=mask_n[:, None] & mask_k[None, :],
                other=0.0,
                eviction_policy='evict_first',
            )

            # dot: x_tile (M,K) * w_tile^T (K,N) -> (M,N)
            logits_tile += tl.dot(x_tile, tl.trans(w_tile))

        # --- online softmax update for this class block ---
        m_local = tl.max(logits_tile, axis=1)                     # (BLOCK_M,)
        m_new = tl.maximum(m, m_local)
        # shift for numerical stability
        exp_shifted = tl.exp(logits_tile - m_new[:, None])        # (BLOCK_M, BLOCK_N)
        sum_local = tl.sum(exp_shifted, axis=1)                   # (BLOCK_M,)
        d_new = d * tl.exp(m - m_new) + sum_local
        m = m_new
        d = d_new

        # --- capture target logit if target falls in this class block ---
        # mask for rows where target is in [class_start, class_start+BLOCK_N)
        target_in_block = (targets >= class_start) & (targets < class_start + BLOCK_N)
        idx_in_block = targets - class_start                     # (BLOCK_M,)
        # build one-hot mask (BLOCK_M, BLOCK_N)
        mask_target = target_in_block[:, None] & (tl.arange(0, BLOCK_N)[None, :] == idx_in_block[:, None])
        # extract logit for each row
        logit_from_block = tl.sum(logits_tile * tl.cast(mask_target, tl.float32), axis=1)
        logit_target = tl.where(target_in_block, logit_from_block, logit_target)

    # final loss per row: logsumexp - target_logit
    loss = (m + tl.log(d)) - logit_target
    tl.store(loss_per_row_ptr + offsets_m, loss, mask=mask_m, eviction_policy='evict_last')


# ---------------------------------------------------------------------------
# Kernel 2: reduce per-row losses to mean.
# Uses a single block with a loop; sufficient for 4096 rows.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_mean_kernel(
    inp_ptr,            # f32 [R]
    out_scalar_ptr,     # f32 [1]
    R: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    total = tl.zeros([], dtype=tl.float32)
    for start in range(0, R, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < R
        vals = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
        total += tl.sum(vals)
    mean = total / R
    tl.store(out_scalar_ptr, mean)


def triton_run(x: torch.Tensor, w: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    R, K = x.shape
    N, K2 = w.shape
    assert K == K2, "hidden dims must match"
    assert targets.shape == (R,), "targets must match batch size"
    device = x.device

    # Tuning parameters - reduced block sizes to fit within shared memory limit (101376 bytes)
    BLOCK_M = 32
    BLOCK_N = 64
    BLOCK_K = 64
    num_warps = 4
    num_stages = 2  # reduced stages to lower shared memory usage

    # intermediate per-row loss buffer
    loss_per_row = torch.empty(R, dtype=torch.float32, device=device)
    # final scalar output
    scalar_out = torch.empty(1, dtype=torch.float32, device=device)

    # launch kernel 1 (grid over row tiles)
    grid_m = triton.cdiv(R, BLOCK_M)
    fused_linear_ce_kernel[(grid_m,)](
        x, w, targets, loss_per_row,
        R, N, K,
        BLOCK_M, BLOCK_N, BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    # launch kernel 2 (reduction to mean)
    # Use a large block size to cover all rows in few loop iterations
    reduce_mean_kernel[(1,)](
        loss_per_row, scalar_out,
        R=R, BLOCK_SIZE=4096,
        num_warps=8,
        num_stages=4,
    )

    return scalar_out  # scalar tensor