import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Transpose kernel: w [N, K] -> w_T [K, N]
# 2D block (BLOCK_T, BLOCK_T)
# ---------------------------------------------------------------------------
@triton.jit
def transpose_kernel(
    inp_ptr,        # [N, K] row-major
    out_ptr,        # [K, N] row-major
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks_n = tl.cdiv(N, BLOCK)
    block_id_n = pid % num_blocks_n
    block_id_k = pid // num_blocks_n
    n_start = block_id_n * BLOCK
    k_start = block_id_k * BLOCK
    n_offsets = tl.arange(0, BLOCK)
    k_offsets = tl.arange(0, BLOCK)
    n_mask = (n_start + n_offsets) < N
    k_mask = (k_start + k_offsets) < K
    mask = n_mask[:, None] & k_mask[None, :]
    # Load tile from inp (N, K)
    inp_ptrs = inp_ptr + (n_start + n_offsets[:, None]) * K + (k_start + k_offsets[None, :])
    tile = tl.load(inp_ptrs, mask=mask, other=0.0)
    # Store transposed tile into out (K, N)
    out_ptrs = out_ptr + (k_start + k_offsets[:, None]) * N + (n_start + n_offsets[None, :])
    tl.store(out_ptrs, tile, mask=mask)


# ---------------------------------------------------------------------------
# Fused linear + cross‑entropy per row.
# For each row m, we compute logits = x[m,:] @ w_T (w_T is [K, N])
# in tiles along N, and perform online softmax + NLL loss for the target.
# ---------------------------------------------------------------------------
@triton.jit
def fused_ce_row_kernel(
    x_ptr,          # [M, K]
    w_T_ptr,        # [K, N] (transposed)
    targets_ptr,    # [M] (int64)
    loss_ptr,       # [M] output per-row loss (float32)
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)               # row index
    target = tl.load(targets_ptr + pid)

    # Online softmax state (all scalars)
    m_old = tl.full([], float('-inf'), dtype=tl.float32)
    d_old = tl.full([], 0.0, dtype=tl.float32)
    target_logit = tl.full([], 0.0, dtype=tl.float32)

    # Outer loop over N tiles
    for n_start in range(0, N, BLOCK_N):
        # Accumulator for this N tile (will hold logits for the tile)
        acc = tl.zeros([1, BLOCK_N], dtype=tl.float32)
        # Inner loop over K (reduction dimension)
        for k_start in range(0, K, BLOCK_K):
            # Load a segment of the x row: shape (BLOCK_K,)
            x_offsets = k_start + tl.arange(0, BLOCK_K)
            x_vals = tl.load(
                x_ptr + pid * K + x_offsets,
                mask=x_offsets < K,   # K is exact, but for safety
                other=0.0,
                eviction_policy='evict_first',
            )
            # Load a 2D tile from w_T: shape (BLOCK_K, BLOCK_N)
            k_off = k_start + tl.arange(0, BLOCK_K)[:, None]
            n_off = n_start + tl.arange(0, BLOCK_N)[None, :]
            w_ptrs = w_T_ptr + k_off * N + n_off
            w_tile = tl.load(
                w_ptrs,
                mask=(k_off < K) & (n_off < N),  # both exact, but for safety
                other=0.0,
                eviction_policy='evict_first',
            )
            # Dot product: (1, BLOCK_K) @ (BLOCK_K, BLOCK_N) -> (1, BLOCK_N)
            x_row = x_vals[None, :]
            acc += tl.dot(x_row, w_tile)

        # acc now contains all logits for this N tile (shape [1, BLOCK_N])
        # Online softmax update – use scalar reductions everywhere
        m_loc = tl.max(acc)                 # scalar
        m_new = tl.maximum(m_old, m_loc)
        exp_centered = tl.exp(acc - m_new)
        sum_exp = tl.sum(exp_centered)      # scalar
        d_new = d_old * tl.exp(m_old - m_new) + sum_exp
        m_old = m_new
        d_old = d_new

        # Extract target logit if the target index belongs to this tile
        target_in_tile = (target >= n_start) & (target < n_start + BLOCK_N)
        col_offs = target - n_start
        col_mask = tl.arange(0, BLOCK_N) == col_offs   # shape (BLOCK_N,)
        extracted = tl.sum(acc * col_mask[None, :])      # scalar (only one non-zero element)
        target_logit = tl.where(target_in_tile, extracted, target_logit)

    # Final logsumexp and loss
    logsumexp = m_old + tl.log(d_old)
    loss = logsumexp - target_logit
    tl.store(loss_ptr + pid, loss, eviction_policy='evict_last')


# ---------------------------------------------------------------------------
# First reduction stage: sum over blocks of rows
# ---------------------------------------------------------------------------
@triton.jit
def reduce_sum_stage1_kernel(
    inp_ptr,          # [R]
    out_partial_ptr,  # [R / BLOCK]
    R: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * BLOCK
    offs = start + tl.arange(0, BLOCK)
    mask = offs < R
    vals = tl.load(inp_ptr + offs, mask=mask, other=0.0)
    partial = tl.sum(vals, axis=0)
    tl.store(out_partial_ptr + pid, partial)


# ---------------------------------------------------------------------------
# Second reduction stage: sum all partials and divide by R (mean)
# ---------------------------------------------------------------------------
@triton.jit
def reduce_mean_stage2_kernel(
    inp_ptr,          # [num_partials]
    out_scalar_ptr,   # [1]
    R: tl.constexpr,           # total rows (for division)
    num_partials: tl.constexpr,
    BLOCK: tl.constexpr,
):
    total = tl.zeros([], dtype=tl.float32)
    for start in range(0, num_partials, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < num_partials
        vals = tl.load(inp_ptr + offs, mask=mask, other=0.0)
        total += tl.sum(vals, axis=0)
    tl.store(out_scalar_ptr, total / R)


# ---------------------------------------------------------------------------
# Main function: triton_run(x, w, targets) -> scalar loss
# ---------------------------------------------------------------------------
def triton_run(x, w, targets) -> torch.Tensor:
    M, K = x.shape
    N, _ = w.shape          # N = 32768, K = 2048
    device = x.device
    dtype = x.dtype

    # ----- Transpose w to [K, N] for efficient dot product -----
    w_T = torch.empty(K, N, device=device, dtype=dtype)
    BLOCK_T = 32
    grid_t = (triton.cdiv(N, BLOCK_T) * triton.cdiv(K, BLOCK_T),)
    transpose_kernel[grid_t](
        w, w_T, N, K, BLOCK_T,
        num_warps=4,
        num_stages=3,
    )

    # ----- Per-row fused linear + cross‑entropy -----
    loss_row = torch.empty(M, device=device, dtype=torch.float32)
    BLOCK_N = 1024    # divides N exactly (32768 / 1024 = 32)
    BLOCK_K = 256     # divides K exactly (2048 / 256 = 8)
    grid_row = (M,)
    fused_ce_row_kernel[grid_row](
        x, w_T, targets, loss_row,
        M, N, K,
        BLOCK_N, BLOCK_K,
        num_warps=8,
        num_stages=4,
    )

    # ----- Reduction to mean -----
    BLOCK_REDUCE = 1024
    num_partials = (M + BLOCK_REDUCE - 1) // BLOCK_REDUCE
    partials = torch.empty(num_partials, device=device, dtype=torch.float32)
    reduce_sum_stage1_kernel[(num_partials,)](
        loss_row, partials, M, BLOCK_REDUCE,
        num_warps=4,
        num_stages=3,
    )

    scalar = torch.empty(1, device=device, dtype=torch.float32)
    reduce_mean_stage2_kernel[(1,)](
        partials, scalar, M, num_partials, BLOCK_REDUCE,
        num_warps=4,
        num_stages=3,
    )

    return scalar