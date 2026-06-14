import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: processes a tile of rows.
#  - For each row, computes the cross-entropy loss via online softmax.
#  - Accumulates the sum of losses for the tile.
#  - Writes a single partial sum to global memory.
# ---------------------------------------------------------------------------
@triton.jit
def multirow_ce_kernel(
    logits_ptr,           # f32 [R, N]
    targets_ptr,          # i64 [R]
    partials_ptr,         # f32 [num_partials]
    R: tl.constexpr,
    N: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    tile_start = pid * ROWS_PER_PROG

    total_loss = tl.zeros([], dtype=tl.float32)

    # Pre-compute strides for the tile – ROWS_PER_PROG is small enough to unroll
    for r in range(ROWS_PER_PROG):
        row = tile_start + r
        target = tl.load(targets_ptr + row)

        # Online softmax state
        m_curr = tl.full([], float('-inf'), dtype=tl.float32)
        d_curr = tl.full([], 0.0, dtype=tl.float32)

        row_base = row * N
        for start in range(0, N, BLOCK_N):
            offs = start + tl.arange(0, BLOCK_N)
            x = tl.load(
                logits_ptr + row_base + offs,
                mask=offs < N,
                other=float('-inf'),
                eviction_policy='evict_first',
            )

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
        total_loss += loss

    tl.store(partials_ptr + pid, total_loss, eviction_policy='evict_last')


# ---------------------------------------------------------------------------
# Kernel 2: reduces the partial sums into a scalar mean.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_mean_kernel(
    partials_ptr,      # f32 [num_partials]
    out_ptr,           # f32 []
    num_partials: tl.constexpr,
    BLOCK_R: tl.constexpr,
    INV_R: tl.constexpr,   # 1.0 / R (float)
):
    offs = tl.arange(0, BLOCK_R)
    mask = offs < num_partials
    vals = tl.load(partials_ptr + offs, mask=mask, other=0.0,
                   eviction_policy='evict_first')
    total = tl.sum(vals, axis=0)
    mean = total * INV_R
    tl.store(out_ptr, mean)


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

    # Tuned parameters (powers of two; N and R are multiples of these)
    BLOCK_N = 4096            # 32768 / 4096 = 8 iterations
    ROWS_PER_PROG = 64        # 8192 / 64 = 128 tiles
    NUM_WARPS_CE = 8          # 256 threads per program for the CE kernel
    NUM_WARPS_REDUCE = 4      # 128 threads for the reduction (block of 128)
    BLOCK_REDUCE = 128        # exactly matches num_partials (128)

    num_partials = R // ROWS_PER_PROG   # 128
    assert num_partials * ROWS_PER_PROG == R, "ROWS_PER_PROG must divide R"

    # Allocate partial sums and output
    partials = torch.empty(num_partials, dtype=torch.float32, device=logits.device)
    out = torch.empty((), dtype=torch.float32, device=logits.device)

    inv_R = 1.0 / float(R)
    grid_ce = (num_partials,)

    # Step 1: multi-row CE + partial sum accumulation
    multirow_ce_kernel[grid_ce](
        logits, targets, partials,
        R, N,
        ROWS_PER_PROG, BLOCK_N,
        num_warps=NUM_WARPS_CE,
    )

    # Step 2: final reduction (sum & divide by R)
    reduce_mean_kernel[(1,)](
        partials, out,
        num_partials, BLOCK_REDUCE,
        INV_R=inv_R,
        num_warps=NUM_WARPS_REDUCE,
    )

    return out