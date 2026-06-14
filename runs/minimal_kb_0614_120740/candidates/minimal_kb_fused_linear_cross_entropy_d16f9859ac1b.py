import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Transpose kernel: w [N, K] -> w_T [K, N]
# 2D block (16,16) is typical; we use BLOCK=32 for better throughput.
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
    # Number of blocks along each dimension
    num_blocks_n = tl.cdiv(N, BLOCK)
    block_id_n = pid % num_blocks_n
    block_id_k = pid // num_blocks_n
    n_start = block_id_n * BLOCK
    k_start = block_id_k * BLOCK
    n_offsets = tl.arange(0, BLOCK)
    k_offsets = tl.arange(0, BLOCK)
    # Mask for boundaries
    n_mask = (n_start + n_offsets) < N
    k_mask = (k_start + k_offsets) < K
    mask = n_mask[:, None] & k_mask[None, :]
    # Load tile from inp (N,K)
    inp_ptrs = inp_ptr + (n_start + n_offsets[:, None]) * K + (k_start + k_offsets[None, :])
    tile = tl.load(inp_ptrs, mask=mask, other=0.0)
    # Store transposed tile into out (K,N)
    out_ptrs = out_ptr + (k_start + k_offsets[:, None]) * N + (n_start + n_offsets[None, :])
    tl.store(out_ptrs, tile, mask=mask)


# ---------------------------------------------------------------------------
# Fused linear + cross‑entropy per row.
# For each row m, we compute logits = x[m,:] @ w_T (w_T is [K,N])
# using manual loads and tl.dot with reshaped x.
# Online softmax + NLL extraction.
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
    row_x_base = pid * K

    # Online softmax state
    m_old = tl.full([], float('-inf'), dtype=tl.float32)
    d_old = tl.full([], 0.0, dtype=tl.float32)
    target_logit = tl.full([], 0.0, dtype=tl.float32)

    # Outer loop over N tiles
    for n_start in range(0, N, BLOCK_N):
        # Accumulator for this N tile
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)

        # Inner loop over K (reduction dimension)
        for k_start in range(0, K, BLOCK_K):
            # Load x row segment (1D vector of length BLOCK_K)
            x_offsets = k_start + tl.arange(0, BLOCK_K)
            x_mask = x_offsets < K
            x_seg = tl.load(x_ptr + row_x_base + x_offsets, mask=x_mask, other=0.0)
            # Reshape to 2D for tl.dot: (1, BLOCK_K)
            x_2d = tl.reshape(x_seg, (1, BLOCK_K))

            # Load w_T tile: shape (BLOCK_K, BLOCK_N)
            w_offsets_n = n_start + tl.arange(0, BLOCK_N)
            w_mask_n = w_offsets_n < N
            # We load element by element? Better to load 2D tile using pointer arithmetic.
            # Use contiguous load with stride N for w_T (K rows, N columns)
            w_2d = tl.zeros([BLOCK_K, BLOCK_N], dtype=tl.float32)
            for kk in range(BLOCK_K):
                row = k_start + kk
                if row < K:
                    # Load one row of w_T (size BLOCK_N)
                    w_row_ptrs = w_T_ptr + row * N + w_offsets_n
                    w_row = tl.load(w_row_ptrs, mask=w_mask_n, other=0.0)
                    w_2d = tl.store(w_2d, w_row, mask=None)  # not valid; we need to assign rows
                    # Actually we need to fill w_2d row by row.
            # This is inefficient. Better to use block pointers or a single 2D load.
            # Since we avoid block pointers, we can load w_T as a 2D block using a loop over rows.
            # But that's too many loads. Alternative: use tl.dot with manually built tiles.
            # However, for simplicity and correctness, we can use the original block pointer approach but fix the error.
            # Given that the error is "input and other must have equal ranks >= 2", it might be a version bug.
            # Let's try using tl.dot with 2D tensors obtained via tl.load with 2D offsets.
        # For now, we break and use a simpler approach: directly load w_T as a 2D block using tl.load with 2D pointer.
    # This is getting messy. Let's revert to block pointers but ensure they work.

# ---------------------------------------------------------------------------
# After analysis, the error "input and other must have equal ranks >= 2" likely arises
# from using block pointers inside a dynamic loop. To quickly fix, we use manual loads
# as in the reference cross_entropy kernel, but that leads to a slow row-by-row load of w_T.
# Instead, we keep the block pointer approach but make the K loop static by using
# tl.static_range? Actually tl.static_range is not valid. However, we can unroll with
# a for loop that uses tl.cdiv? Not possible.
# The safest fix: keep the original kernel but replace tl.dot with manual sum of products.
# That is: for each kk in BLOCK_K, load a scalar of x and a vector of w_T, accumulate.
# This is slower but correct. Since we are in round 1 and need correctness, we sacrifice speed.
# We can later optimize with tl.dot if we find a solution.
#
# Let's implement the manual sum-of-products version.
# ---------------------------------------------------------------------------
@triton.jit
def fused_ce_row_kernel_v2(
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
    pid = tl.program_id(0)
    target = tl.load(targets_ptr + pid)
    row_x_base = pid * K

    m_old = tl.full([], float('-inf'), dtype=tl.float32)
    d_old = tl.full([], 0.0, dtype=tl.float32)
    target_logit = tl.full([], 0.0, dtype=tl.float32)

    for n_start in range(0, N, BLOCK_N):
        # Initialize accumulator for this N tile
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        # Loop over K
        for k_start in range(0, K, BLOCK_K):
            # Load x row segment (scalar elements? We'll load one element at a time)
            # To avoid register pressure, we load a small vector of x and corresponding w_T slice.
            # Actually, we need to compute dot product: for each kk in BLOCK_K, add x[kk] * w_T[kk, :].
            # We'll load x segment as vector and w_T segment as matrix and do element-wise multiply and sum.
            x_offsets = k_start + tl.arange(0, BLOCK_K)
            x_mask = x_offsets < K
            x_seg = tl.load(x_ptr + row_x_base + x_offsets, mask=x_mask, other=0.0)  # (BLOCK_K,)

            # Load w_T segment: shape (BLOCK_K, BLOCK_N)
            w_offsets_n = n_start + tl.arange(0, BLOCK_N)
            w_mask_n = w_offsets_n < N
            # We need 2D loading. We'll loop over rows of w_T within BLOCK_K.
            # But that would be a loop inside a loop. Instead, we can load the entire w_T tile using a 2D pointer.
            # Since we avoid block pointers, we can use a single tl.load with a 2D offset.
            # Let's compute the base pointer for this w_T tile.
            w_base = w_T_ptr + k_start * N + n_start
            # We need a 2D mask: (BLOCK_K, BLOCK_N)
            row_offsets = tl.arange(0, BLOCK_K)[:, None]
            col_offsets = tl.arange(0, BLOCK_N)[None, :]
            masks = (k_start + row_offsets < K) & (n_start + col_offsets < N)
            w_tile = tl.load(w_base + row_offsets * N + col_offsets, mask=masks, other=0.0)
            # Now compute dot product: acc = acc + sum_k x_seg[k] * w_tile[k, :]
            # We can do: x_seg_broadcast = x_seg[:, None] (BLOCK_K, 1), then multiply and sum over k.
            prod = tl.sum(x_seg[:, None] * w_tile, axis=0)  # (BLOCK_N,)
            acc += prod

        # Online softmax update
        m_loc = tl.max(acc, axis=0)
        m_new = tl.maximum(m_old, m_loc)
        exp_centered = tl.exp(acc - m_new)
        sum_exp = tl.sum(exp_centered, axis=0)
        d_new = d_old * tl.exp(m_old - m_new) + sum_exp
        m_old = m_new
        d_old = d_new

        # Extract target logit if in this tile
        if n_start <= target < n_start + BLOCK_N:
            col_offset = target - n_start
            col_offsets = tl.arange(0, BLOCK_N)
            target_mask = col_offsets == col_offset
            target_logit = tl.sum(acc * target_mask)

    logsumexp = m_old + tl.log(d_old)
    loss = logsumexp - target_logit
    tl.store(loss_ptr + pid, loss)


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
    fused_ce_row_kernel_v2[grid_row](
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