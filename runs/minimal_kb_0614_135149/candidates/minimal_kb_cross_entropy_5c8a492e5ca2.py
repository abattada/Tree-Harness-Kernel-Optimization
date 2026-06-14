import torch
import triton
import triton.language as tl


@triton.jit
def fused_row_reduce_kernel(
    logits_ptr,      # float32 [R, N]
    targets_ptr,     # int64 [R]
    partials_ptr,    # float32 [grid_size]  – per-program partial sums
    R: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    """
    Each program processes ROWS_PER_PROG consecutive rows, computes the
    per-row cross-entropy loss using online softmax, and accumulates the
    results into a single partial sum.  This fuses row-wise loss computation
    and the first reduction stage, eliminating the intermediate loss_row buffer.
    """
    pid = tl.program_id(0)
    # With exact division, each program handles exactly ROWS_PER_PROG rows.
    tl.static_assert(R % ROWS_PER_PROG == 0)
    row_start = pid * ROWS_PER_PROG

    group_sum = tl.zeros([], dtype=tl.float32)

    for r in range(ROWS_PER_PROG):
        row_idx = row_start + r
        target = tl.load(targets_ptr + row_idx, eviction_policy='evict_first')

        # Initialise online softmax state for this row.
        m_curr = tl.full([], float('-inf'), dtype=tl.float32)
        d_curr = tl.full([], 0.0, dtype=tl.float32)

        row_base = row_idx * N
        tl.static_assert(N % BLOCK_SIZE_N == 0)

        for start in range(0, N, BLOCK_SIZE_N):
            offs = start + tl.arange(0, BLOCK_SIZE_N)
            # No mask: BLOCK_SIZE_N divides N, so offs stays in bounds.
            x = tl.load(logits_ptr + row_base + offs,
                        eviction_policy='evict_first')

            m_loc = tl.max(x, axis=0)
            m_new = tl.maximum(m_curr, m_loc)
            exp_centered = tl.exp(x - m_new)
            sum_exp = tl.sum(exp_centered, axis=0)
            d_new = d_curr * tl.exp(m_curr - m_new) + sum_exp
            m_curr = m_new
            d_curr = d_new

        logsumexp = m_curr + tl.log(d_curr)
        target_logit = tl.load(logits_ptr + row_base + target)
        loss = logsumexp - target_logit
        group_sum += loss

    # Write the partial sum for this program.
    tl.store(partials_ptr + pid, group_sum, eviction_policy='evict_last')


@triton.jit
def reduce_mean_final_kernel(
    partials_ptr,    # float32 [num_partials]
    out_ptr,         # float32 []  scalar output
    R: tl.constexpr,
    num_partials: tl.constexpr,
    BLOCK_SIZE_RED: tl.constexpr,
):
    """
    Single‑block reduction: sums all partial sums and divides by the number
    of rows (R) to produce the final mean cross‑entropy.
    """
    offs = tl.arange(0, BLOCK_SIZE_RED)
    mask = offs < num_partials
    vals = tl.load(partials_ptr + offs, mask=mask, other=0.0)
    total = tl.sum(vals, axis=0)
    mean = total / tl.full([], R, dtype=tl.float32)
    if tl.program_id(0) == 0:
        tl.store(out_ptr, mean)


def triton_run(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits : float32 [8192, 32768]
    targets: int64   [8192]
    returns: float32 scalar = mean cross‑entropy loss
    """
    R, N = logits.shape
    assert targets.shape == (R,)
    assert logits.dtype == torch.float32
    assert targets.dtype == torch.int64

    # Tuned parameters for RTX 5090
    BLOCK_SIZE_N = 1024          # divides N=32768
    ROWS_PER_PROG = 64           # divides R=8192 → grid_size=128
    NUM_WARPS_FUSED = 8          # 256 threads, 4 elements/thread per chunk
    BLOCK_SIZE_RED = 256         # must be ≥ num_partials (128)
    NUM_WARPS_FINAL = 4

    grid_size = R // ROWS_PER_PROG   # 128 programs
    num_partials = grid_size

    partials = torch.empty(num_partials, dtype=torch.float32, device=logits.device)

    # Fused kernel: per‑row loss + first‑stage reduction → partials
    fused_row_reduce_kernel[(grid_size,)](
        logits, targets, partials,
        R=R, N=N, BLOCK_SIZE_N=BLOCK_SIZE_N, ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=NUM_WARPS_FUSED,
    )

    out = torch.empty((), dtype=torch.float32, device=logits.device)

    # Second‑stage reduction: sum partials → scalar mean
    reduce_mean_final_kernel[(1,)](
        partials, out,
        R=R, num_partials=num_partials, BLOCK_SIZE_RED=BLOCK_SIZE_RED,
        num_warps=NUM_WARPS_FINAL,
    )

    return out