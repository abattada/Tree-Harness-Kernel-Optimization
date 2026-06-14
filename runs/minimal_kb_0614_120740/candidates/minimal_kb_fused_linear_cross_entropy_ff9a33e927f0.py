import torch
import triton
import triton.language as tl
from math import prod

# ---------------------------------------------------------------------------
# Fused linear + cross-entropy kernel.
# Each program processes a tile of rows (BLOCK_M) and tiles over classes (N)
# and features (K).  We compute logits tiles on the fly, apply online softmax,
# then compute the per‑row loss and reduce to a partial sum for this program.
# ---------------------------------------------------------------------------
@triton.jit
def fused_linear_ce_kernel(
    x_ptr,          # [M, K]    in row‑major
    w_t_ptr,        # [K, N]    pre‑transposed row‑major (contiguous along K)
    targets_ptr,    # [M]       long
    partial_loss_ptr,  # [num_programs] output partial sums
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    # we handle remainder by masking inside loops; no early exit needed

    # Shared memory allocations (dynamic)
    x_shared = tl.tl_extra_shared_memory(0)   # [BLOCK_M, BLOCK_K] floats
    w_shared = tl.tl_extra_shared_memory(1)   # [BLOCK_K, BLOCK_N] floats
    logits_shared = tl.tl_extra_shared_memory(2) # [BLOCK_M, BLOCK_N] floats

    # Per‑row online softmax state
    m_old = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    d_old = tl.full([BLOCK_M], 0.0, dtype=tl.float32)

    # Loop over N tiles
    for n_start in range(0, N, BLOCK_N):
        # Zero out the logits accumulator for this N tile
        # We'll store in shared memory as 2D array
        # Use a loop over rows to avoid massive register usage
        for r in range(BLOCK_M):
            # Clear row r of logits_shared
            offs_n = tl.arange(0, BLOCK_N)
            tl.store(logits_shared + r * BLOCK_N + offs_n, tl.zeros([BLOCK_N], dtype=tl.float32))

        # K tile reduction loop
        for k_start in range(0, K, BLOCK_K):
            # Load x tile [BLOCK_M, BLOCK_K] from global to shared
            # Shape: (BLOCK_M, BLOCK_K) in row‑major, contiguous along K
            # offsets: x_ptr[row_start + r, k_start + kk]
            # We'll load in a loop over rows to stay within register limits
            # But better to load directly with masks using tl.load (vectorised)
            # Since we are loading a 2D tile, we can use a single load with proper
            # multi‑dimensional indirection.  Triton supports 2D block loads.
            # We'll use a loop over rows for simplicity.
            for r in range(BLOCK_M):
                row_global = row_start + r
                if row_global < M:
                    offs_k = k_start + tl.arange(0, BLOCK_K)
                    mask_k = offs_k < K
                    x_vals = tl.load(x_ptr + row_global * K + offs_k, mask=mask_k, other=0.0)
                else:
                    x_vals = tl.zeros([BLOCK_K], dtype=tl.float32)
                tl.store(x_shared + r * BLOCK_K + tl.arange(0, BLOCK_K), x_vals)

            # Load w_t tile [BLOCK_K, BLOCK_N] from global to shared
            # w_t is stored as [K, N] row‑major (contiguous along K for each N)
            # So for a given n_start and k_start, we load a rectangular region of size (BLOCK_K, BLOCK_N)
            # without transpose.
            for kk in range(BLOCK_K):
                k_global = k_start + kk
                if k_global < K:
                    offs_n = n_start + tl.arange(0, BLOCK_N)
                    mask_n = offs_n < N
                    w_vals = tl.load(w_t_ptr + k_global * N + offs_n, mask=mask_n, other=0.0)
                else:
                    w_vals = tl.zeros([BLOCK_N], dtype=tl.float32)
                tl.store(w_shared + kk * BLOCK_N + tl.arange(0, BLOCK_N), w_vals)

            # Synchronize to ensure shared memory is ready (not needed inside async copy? We'll use tl.debug_barrier)
            tl.debug_barrier()

            # Compute dot product: x * w_t  => (BLOCK_M, BLOCK_N) partial
            # We'll load x and w from shared and accumulate into registers, then store.
            # Use tl.dot directly – it requires operands in registers, but we can load from shared.
            # Load x tile from shared: shape (BLOCK_M, BLOCK_K)
            x_vals = tl.load(x_shared + (tl.arange(0, BLOCK_M)[:, None] * BLOCK_K +
                                         tl.arange(0, BLOCK_K)[None, :]))
            # Load w tile from shared: shape (BLOCK_K, BLOCK_N)
            w_vals = tl.load(w_shared + (tl.arange(0, BLOCK_K)[:, None] * BLOCK_N +
                                         tl.arange(0, BLOCK_N)[None, :]))
            # Accumulate into logits_shared (which we will keep in registers? No, store back)
            # Compute dot product
            dot = tl.dot(x_vals, w_vals)  # returns (BLOCK_M, BLOCK_N) float32
            # Accumulate into shared memory logits tile
            # We need to add element‑wise.  Since we cannot atomics, we load current,
            # add, store.  We'll do it row‑wise.
            for r in range(BLOCK_M):
                offs_n = tl.arange(0, BLOCK_N)
                current = tl.load(logits_shared + r * BLOCK_N + offs_n)
                new_vals = current + dot[r, :]
                tl.store(logits_shared + r * BLOCK_N + offs_n, new_vals)

            tl.debug_barrier()

        # At this point logits for this N tile are in logits_shared.
        # Apply online softmax per row.
        for r in range(BLOCK_M):
            row_global = row_start + r
            if row_global >= M:
                continue
            # Load logits for this row (BLOCK_N values)
            offs_n = tl.arange(0, BLOCK_N)
            logits = tl.load(logits_shared + r * BLOCK_N + offs_n)
            # Online softmax update
            m_loc = tl.max(logits, axis=0)
            m_new = tl.maximum(m_old[r], m_loc)
            # Because we might have NaNs from inf inputs? Ensure finite.
            exp_centered = tl.exp(logits - m_new)
            sum_exp = tl.sum(exp_centered, axis=0)
            # Guard against overflow: d_new may become inf if m_old is -inf but exp(m_old-m_new) is zero? Actually exp(-inf) = 0, fine.
            d_new = d_old[r] * tl.exp(m_old[r] - m_new) + sum_exp
            # Update state
            m_old[r] = m_new
            d_old[r] = d_new

    # After all N tiles, compute logsumexp per row
    logsumexp = m_old + tl.log(d_old)

    # Compute target logit for each row via a separate K reduction
    loss_partial = tl.zeros([BLOCK_M], dtype=tl.float32)
    for r in range(BLOCK_M):
        row_global = row_start + r
        if row_global >= M:
            continue
        # Load x row
        # We'll loop over K in BLOCK_K chunks to reduce register usage
        target = tl.load(targets_ptr + row_global).to(tl.int32)
        if target >= N:
            # safety, though targets are valid
            loss_val = 0.0
        else:
            target_logit = tl.zeros([], dtype=tl.float32)
            for k_start in range(0, K, BLOCK_K):
                offs_k = k_start + tl.arange(0, BLOCK_K)
                mask_k = offs_k < K
                x_vals = tl.load(x_ptr + row_global * K + offs_k, mask=mask_k, other=0.0)
                # Load w[target, k_start: k_start+BLOCK_K] – note w is stored as [N, K] (original).
                # But we have w_t transposed: w_t[(k_start..), target] is the column.
                # Actually w_t is [K, N], so we need to load a column: w_t[k_start + kk, target].
                # That's not contiguous.  Better to load a vector along K for a fixed N index
                # by using w_t[k_start + kk, target] which is strided by N.
                # We can load as a gather or loop over kk.  Since BLOCK_K is small, loop.
                w_vals = tl.zeros([BLOCK_K], dtype=tl.float32)
                for kk in range(BLOCK_K):
                    k_global = k_start + kk
                    if k_global < K:
                        w_vals[kk] = tl.load(w_t_ptr + k_global * N + target)
                target_logit += tl.sum(x_vals * w_vals)
            # loss = logsumexp - target_logit
            loss_val = logsumexp[r] - target_logit
        loss_partial[r] = loss_val

    # Sum over rows in this tile, write to partial array
    total_loss = tl.sum(loss_partial, axis=0)
    tl.store(partial_loss_ptr + pid, total_loss)


# ---------------------------------------------------------------------------
# Reduction kernel: sum all partial losses and divide by M.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_mean_kernel(
    partial_ptr,   # [num_partials] float32
    scalar_ptr,    # [1] float32
    num_partials: tl.constexpr,
    M: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    total = tl.zeros([], dtype=tl.float32)
    for start in range(0, num_partials, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < num_partials
        vals = tl.load(partial_ptr + offs, mask=mask, other=0.0)
        total += tl.sum(vals, axis=0)
    tl.store(scalar_ptr, total / M)


def triton_run(x, w, targets) -> torch.Tensor:
    """
    x: (M, K) float32
    w: (N, K) float32
    targets: (M,) int64
    returns scalar float32
    """
    M, K = x.shape
    N, _ = w.shape

    # Pre‑transpose w from [N, K] to [K, N] for faster contiguous access in kernel
    w_t = w.contiguous().t().contiguous()  # (K, N)

    # Tuning parameters (these can be adjusted)
    BLOCK_M = 32
    BLOCK_N = 256
    BLOCK_K = 64

    assert M % BLOCK_M == 0, "M must be divisible by BLOCK_M for this seed implementation"
    num_programs = M // BLOCK_M

    # Partial sums array
    partial = torch.empty(num_programs, dtype=torch.float32, device=x.device)

    # Shared memory sizes (in bytes)
    x_shared_bytes = BLOCK_M * BLOCK_K * 4
    w_shared_bytes = BLOCK_K * BLOCK_N * 4
    logits_shared_bytes = BLOCK_M * BLOCK_N * 4
    max_shared_bytes = max(x_shared_bytes, w_shared_bytes, logits_shared_bytes) * 3  # allocate all three
    # We'll allocate dynamic shared memory of size sum of all three, but Triton allows total.

    grid = (num_programs,)
    fused_linear_ce_kernel[grid](
        x, w_t, targets, partial,
        M, N, K,
        BLOCK_M, BLOCK_N, BLOCK_K,
        num_warps=4,
        num_stages=4,
        shared_memory_size= x_shared_bytes + w_shared_bytes + logits_shared_bytes,
    )

    # Second reduction kernel
    scalar_out = torch.empty(1, dtype=torch.float32, device=x.device)
    reduce_grid = (1,)
    reduce_mean_kernel[reduce_grid](
        partial, scalar_out,
        num_partials=num_programs,
        M=M,
        BLOCK_SIZE=4096,
        num_warps=1,
        num_stages=2,
    )
    return scalar_out