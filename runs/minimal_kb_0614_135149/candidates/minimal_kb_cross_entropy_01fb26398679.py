import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: per‑program processes a chunk of rows, computes per‑row
# cross‑entropy loss (online softmax) and accumulates the sum of losses.
# ---------------------------------------------------------------------------
@triton.jit
def cross_entropy_partial_kernel(
    logits_ptr,        # f32 [R, N]
    targets_ptr,       # i64 [R]
    partial_out_ptr,   # f32 [num_partials]
    R: tl.constexpr,
    N: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)

    # All programs process exactly ROWS_PER_PROGRAM rows
    row_start = pid * ROWS_PER_PROGRAM
    sum_loss = tl.zeros([], dtype=tl.float32)

    for r in range(ROWS_PER_PROGRAM):
        row_idx = row_start + r
        target = tl.load(targets_ptr + row_idx)

        # Online softmax: single-pass max and sum of exp
        m_old = tl.full([], float('-inf'), dtype=tl.float32)
        d_old = tl.full([], 0.0, dtype=tl.float32)

        row_base = row_idx * N
        # N is a multiple of BLOCK_SIZE_N (32768 / 1024 = 32), so no mask needed
        for start in range(0, N, BLOCK_SIZE_N):
            offs = start + tl.arange(0, BLOCK_SIZE_N)
            x = tl.load(
                logits_ptr + row_base + offs,
                eviction_policy='evict_first',
            )
            m_loc = tl.max(x, axis=0)
            m_new = tl.maximum(m_old, m_loc)
            exp_centered = tl.exp(x - m_new)
            sum_exp = tl.sum(exp_centered, axis=0)
            d_new = d_old * tl.exp(m_old - m_new) + sum_exp
            m_old = m_new
            d_old = d_new

        logsumexp = m_old + tl.log(d_old)
        target_logit = tl.load(logits_ptr + row_base + target)
        loss = logsumexp - target_logit
        sum_loss += loss

    tl.store(partial_out_ptr + pid, sum_loss, eviction_policy='evict_last')


# ---------------------------------------------------------------------------
# Kernel 2: final reduction – sums partial sums and computes the mean loss.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_mean_kernel(
    partial_ptr,       # f32 [num_partials]
    out_scalar_ptr,    # f32 [1]
    R: tl.constexpr,
    NUM_PARTIALS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    # NUM_PARTIALS == BLOCK_SIZE by construction, so mask always true
    partials = tl.load(partial_ptr + offs)
    total = tl.sum(partials, axis=0)
    mean = total / tl.full([], R, dtype=tl.float32)
    tl.store(out_scalar_ptr, mean)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def triton_run(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits : float32 [8192, 32768]
    targets: int64   [8192]
    returns: float32 scalar = mean cross-entropy loss
    """
    R, N = logits.shape
    assert targets.shape == (R,)
    assert logits.dtype == torch.float32
    assert targets.dtype == torch.int64

    # Tuned constants specialised for the given shapes
    BLOCK_SIZE_N = 1024        # divides N exactly
    ROWS_PER_PROGRAM = 256     # divides R exactly (8192 / 256 = 32)
    NUM_PARTIALS = R // ROWS_PER_PROGRAM  # = 32
    BLOCK_SIZE_R2 = NUM_PARTIALS         # exact fit

    # Kernel launch configurations
    NUM_WARPS_PARTIAL = 8
    NUM_WARPS_REDUCE = 4

    # Step 1: compute partial sums (one per ROWS_PER_PROGRAM rows)
    partials = torch.empty(NUM_PARTIALS, dtype=torch.float32, device=logits.device)

    grid_partial = (NUM_PARTIALS,)
    cross_entropy_partial_kernel[grid_partial](
        logits, targets, partials,
        R, N, ROWS_PER_PROGRAM, BLOCK_SIZE_N,
        num_warps=NUM_WARPS_PARTIAL,
    )

    # Step 2: reduce partials to a scalar mean
    out = torch.empty((), dtype=torch.float32, device=logits.device)
    reduce_mean_kernel[(1,)](
        partials, out,
        R, NUM_PARTIALS, BLOCK_SIZE_R2,
        num_warps=NUM_WARPS_REDUCE,
    )

    return out