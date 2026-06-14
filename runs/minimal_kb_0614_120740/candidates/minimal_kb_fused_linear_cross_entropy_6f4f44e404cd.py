import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Transpose kernel: w [N, K] -> w_T [K, N]  (2D block)
# ---------------------------------------------------------------------------
@triton.jit
def transpose_kernel(
    inp_ptr,           # [N, K] row-major
    out_ptr,           # [K, N] row-major
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
    inp_ptrs = inp_ptr + (n_start + n_offsets[:, None]) * K + (k_start + k_offsets[None, :])
    tile = tl.load(inp_ptrs, mask=mask, other=0.0)
    out_ptrs = out_ptr + (k_start + k_offsets[:, None]) * N + (n_start + n_offsets[None, :])
    tl.store(out_ptrs, tile, mask=mask)


# ---------------------------------------------------------------------------
# Fused linear + cross‑entropy per row, online softmax.
# Launched with grid = (M,), each program processes one row.
# ---------------------------------------------------------------------------
@triton.jit
def fused_ce_row_kernel(
    x_ptr,          # [M, K]
    w_T_ptr,        # [K, N]
    targets_ptr,    # [M] int64
    loss_row_ptr,   # [M] output per-row loss (float32)
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    target = tl.load(targets_ptr + pid)
    row_x_base = pid * K

    # Online softmax state (scalars)
    m_old = tl.full([], float('-inf'), dtype=tl.float32)
    d_old = tl.full([], 0.0, dtype=tl.float32)
    target_logit = tl.full([], 0.0, dtype=tl.float32)

    # Outer loop over N tiles
    for n_start in range(0, N, BLOCK_N):
        # Accumulate logits for this N tile using tl.dot
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            # Load x row segment (1D vector)
            x_offsets = k_start + tl.arange(0, BLOCK_K)
            x_mask = x_offsets < K
            x_seg = tl.load(
                x_ptr + row_x_base + x_offsets,
                mask=x_mask,
                other=0.0,
                eviction_policy='evict_first',
            )
            # Reshape to (1, BLOCK_K) for dot product
            x_2d = tl.reshape(x_seg, (1, BLOCK_K))

            # Load w_T tile: shape (BLOCK_K, BLOCK_N)
            w_offsets_n = n_start + tl.arange(0, BLOCK_N)
            w_offsets_k = k_start + tl.arange(0, BLOCK_K)
            w_mask = (w_offsets_n[None, :] < N) & (w_offsets_k[:, None] < K)
            w_base = w_T_ptr + k_start * N + n_start
            w_tile = tl.load(
                w_base + w_offsets_k[:, None] * N + w_offsets_n[None, :],
                mask=w_mask,
                other=0.0,
                eviction_policy='evict_first',
            )
            # Accumulate: acc (BLOCK_N) += dot(x_2d (1,BLOCK_K), w_tile (BLOCK_K,BLOCK_N))
            acc += tl.dot(x_2d, w_tile).reshape([BLOCK_N])

        # Online softmax update
        m_loc = tl.max(acc, axis=0)
        m_new = tl.maximum(m_old, m_loc)
        exp_centered = tl.exp(acc - m_new)
        sum_exp = tl.sum(exp_centered, axis=0)
        d_new = d_old * tl.exp(m_old - m_new) + sum_exp
        m_old = m_new
        d_old = d_new

        # Extract target logit if in this tile
        if target >= n_start and target < n_start + BLOCK_N:
            col_offset = target - n_start
            target_logit = tl.sum(acc * (tl.arange(0, BLOCK_N) == col_offset))

    # Compute per‑row loss: logsumexp - target_logit
    logsumexp = m_old + tl.log(d_old)
    loss = logsumexp - target_logit
    tl.store(loss_row_ptr + pid, loss, eviction_policy='evict_last')


# ---------------------------------------------------------------------------
# First reduction stage: sum chunks of per‑row losses
# ---------------------------------------------------------------------------
@triton.jit
def reduce_sum_stage1_kernel(
    inp_ptr,           # [R]
    out_partial_ptr,   # [R // BLOCK_SIZE]
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
# Second reduction stage: sum all partials and divide by R (mean)
# ---------------------------------------------------------------------------
@triton.jit
def reduce_mean_stage2_kernel(
    inp_ptr,            # [num_partials]
    out_scalar_ptr,     # [1]
    R: tl.constexpr,                # total rows
    num_partials: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    total = tl.full([], 0.0, dtype=tl.float32)
    for start in range(0, num_partials, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < num_partials
        vals = tl.load(inp_ptr + offs, mask=mask, other=0.0)
        total += tl.sum(vals)
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

    # ----- Per‑row fused linear + cross‑entropy -----
    loss_row = torch.empty(M, device=device, dtype=torch.float32)
    BLOCK_N = 1024    # divides N = 32768 exactly (32 tiles)
    BLOCK_K = 256     # divides K = 2048 exactly (8 tiles)
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