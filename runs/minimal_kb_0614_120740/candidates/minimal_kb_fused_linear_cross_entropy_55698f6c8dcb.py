import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 1: fused linear + cross-entropy (online softmax) per row.
# Each program handles exactly one row (BLOCK_M=1) to keep shared memory low.
# It loops over class blocks (BLOCK_N) and hidden dimension blocks (BLOCK_K).
# ---------------------------------------------------------------------------
@triton.jit
def fused_linear_ce_row_kernel(
    x_ptr,                # f32 [R, K]
    w_ptr,                # f32 [N, K]
    targets_ptr,          # i64 [R]
    loss_per_row_ptr,     # f32 [R]  output: per-row losses
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)        # row index
    # load target for this row
    target = tl.load(targets_ptr + pid)

    # online softmax state (scalars per row)
    m_old = tl.full((), float('-inf'), dtype=tl.float32)
    d_old = tl.full((), 0.0, dtype=tl.float32)
    logit_target = tl.full((), 0.0, dtype=tl.float32)

    # Loop over class blocks
    for class_start in range(0, N, BLOCK_N):
        offsets_n = class_start + tl.arange(0, BLOCK_N)
        mask_n = offsets_n < N

        # Accumulate logits for this class block (1 x BLOCK_N)
        logits_tile = tl.zeros((1, BLOCK_N), dtype=tl.float32)

        # K-loop over hidden dimension
        for k_start in range(0, K, BLOCK_K):
            offsets_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offsets_k < K

            # load x row tile (1, BLOCK_K)
            x_tile = tl.load(
                x_ptr + pid * K + offsets_k[None, :],
                mask=mask_k[None, :],
                other=0.0,
                eviction_policy='evict_first',
            )

            # load w tile (BLOCK_N, BLOCK_K)
            w_tile = tl.load(
                w_ptr + offsets_n[:, None] * K + offsets_k[None, :],
                mask=mask_n[:, None] & mask_k[None, :],
                other=0.0,
                eviction_policy='evict_first',
            )

            # dot product: (1,BLOCK_K) x (BLOCK_K,BLOCK_N) -> (1,BLOCK_N)
            logits_tile += tl.dot(x_tile, tl.trans(w_tile))

        # --- online softmax update for this class block ---
        m_local = tl.max(logits_tile, axis=1)                 # scalar (1,)
        m_new = tl.maximum(m_old, m_local)
        exp_shifted = tl.exp(logits_tile - m_new)             # (1,BLOCK_N)
        sum_local = tl.sum(exp_shifted, axis=1)               # scalar
        d_new = d_old * tl.exp(m_old - m_new) + sum_local
        m_old = m_new
        d_old = d_new

        # --- capture target logit if target falls in this block ---
        target_in_block = (target >= class_start) & (target < class_start + BLOCK_N)
        # one-hot mask for the target column
        col_mask = tl.arange(0, BLOCK_N) == (target - class_start)
        # combine row_valid (always true for this row) and column mask
        mask_target = tl.full((1, BLOCK_N), 1, dtype=tl.int1) & col_mask[None, :]
        # extract logit
        masked_logit = logits_tile * tl.cast(mask_target, tl.float32)
        logit_from_block = tl.sum(masked_logit, axis=1)       # scalar
        logit_target = tl.where(target_in_block, logit_from_block, logit_target)

    # final loss per row: logsumexp - target_logit
    loss = (m_old + tl.log(d_old)) - logit_target
    tl.store(loss_per_row_ptr + pid, loss, eviction_policy='evict_last')


# ---------------------------------------------------------------------------
# Reduction stage 1: each block sums a chunk of per-row losses.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_sum_stage1_kernel(
    inp_ptr,           # f32 [R]
    out_partial_ptr,   # f32 [num_partials]
    R: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < R
    vals = tl.load(inp_ptr + offs, mask=mask, other=0.0)
    partial = tl.sum(vals, axis=0)
    tl.store(out_partial_ptr + pid, partial)


# ---------------------------------------------------------------------------
# Reduction stage 2: sum all partials and compute mean.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_mean_stage2_kernel(
    inp_ptr,           # f32 [num_partials]
    out_scalar_ptr,    # f32 [1]
    R: tl.constexpr,
    num_partials: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    total = tl.zeros((), dtype=tl.float32)
    for start in range(0, num_partials, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < num_partials
        vals = tl.load(inp_ptr + offs, mask=mask, other=0.0)
        total += tl.sum(vals)
    tl.store(out_scalar_ptr, total / R)


# ---------------------------------------------------------------------------
# triton_run: allocate outputs, launch kernels.
# ---------------------------------------------------------------------------
def triton_run(x: torch.Tensor, w: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    R, K = x.shape
    N, K2 = w.shape
    assert K == K2, "hidden dims must match"
    assert targets.shape == (R,), "targets must match batch size"
    device = x.device

    # Tuning parameters – chosen to keep shared memory within limit (<101376 bytes)
    BLOCK_N = 64
    BLOCK_K = 64
    num_warps_fused = 2
    num_stages_fused = 2

    # intermediate per-row loss buffer
    loss_per_row = torch.empty(R, dtype=torch.float32, device=device)
    # final scalar output
    scalar_out = torch.empty(1, dtype=torch.float32, device=device)

    # Kernel 1: one program per row
    grid_m = R
    fused_linear_ce_row_kernel[(grid_m,)](
        x, w, targets, loss_per_row,
        N, K,
        BLOCK_N, BLOCK_K,
        num_warps=num_warps_fused,
        num_stages=num_stages_fused,
    )

    # Reduction: two-stage to be efficient for 4096 rows
    REDUCE_BLOCK = 512          # rows per stage1 block
    num_partials = triton.cdiv(R, REDUCE_BLOCK)
    partial_sums = torch.empty(num_partials, dtype=torch.float32, device=device)

    reduce_sum_stage1_kernel[(num_partials,)](
        loss_per_row, partial_sums,
        R=R, BLOCK_SIZE=REDUCE_BLOCK,
        num_warps=4,
    )

    reduce_mean_stage2_kernel[(1,)](
        partial_sums, scalar_out,
        R=R, num_partials=num_partials, BLOCK_SIZE=256,
        num_warps=4,
    )

    return scalar_out