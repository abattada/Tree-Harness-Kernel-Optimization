import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 1: multi‑row online softmax + NLL loss with partial sum reduction.
# Each program handles ROWS_PER_PROG rows, accumulating their losses into a
# single partial sum.  This eliminates the separate per‑row buffer and the
# first reduction stage, reducing kernel launches and DRAM traffic.
# ---------------------------------------------------------------------------
@triton.jit
def cross_entropy_multirow_kernel(
    logits_ptr,      # f32 [R, N]
    targets_ptr,     # i64 [R]
    partial_sum_ptr, # f32 [num_blocks]   one partial sum per block
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,       # chunk size along the class dimension
    ROWS_PER_PROG: tl.constexpr,    # rows processed by each program
    R: tl.constexpr,                # total rows (only needed for bounds)
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROG
    row_end = tl.minimum(row_start + ROWS_PER_PROG, R)

    accum = tl.zeros([], dtype=tl.float32)

    for row_idx in range(row_start, row_end):
        target = tl.load(targets_ptr + row_idx)
        m_old = tl.full([], float('-inf'), dtype=tl.float32)
        d_old = tl.full([], 0.0, dtype=tl.float32)

        row_base = row_idx * N
        # N is a multiple of BLOCK_SIZE by construction.
        # Therefore the loop always stays inside bounds – we omit the mask.
        for start in range(0, N, BLOCK_SIZE):
            offs = start + tl.arange(0, BLOCK_SIZE)
            x = tl.load(logits_ptr + row_base + offs,
                        eviction_policy='evict_first')

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
        accum += loss

    tl.store(partial_sum_ptr + pid, accum)


# ---------------------------------------------------------------------------
# Kernel 2: final reduction of the partial sums into the mean loss.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_mean_kernel(
    partial_sum_ptr,  # f32 [num_partials]
    out_ptr,          # f32 [1]
    R: tl.constexpr,
    NUM_PARTIALS: tl.constexpr,
):
    offs = tl.arange(0, NUM_PARTIALS)
    # Load all partials; the launch block size must cover NUM_PARTIALS.
    vals = tl.load(partial_sum_ptr + offs)
    total = tl.sum(vals, axis=0)
    mean = total / R
    tl.store(out_ptr, mean)


# ---------------------------------------------------------------------------
# Main entry point – allocates buffers, launches kernels, returns scalar loss.
# ---------------------------------------------------------------------------
def triton_run(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute mean cross-entropy loss.

    Args:
        logits:  float32 tensor of shape [R, N]  (8192, 32768)
        targets: int64 tensor of shape [R]        (8192,)

    Returns:
        scalar float32 tensor with the mean loss.
    """
    R, N = logits.shape
    assert logits.dtype == torch.float32 and targets.dtype == torch.int64

    # Tuned parameters for RTX 5090:
    # - BLOCK_SIZE divides N exactly and reduces loop iterations
    # - ROWS_PER_PROG balances occupancy with block count
    BLOCK_SIZE_CLASS = 8192     # N=32768 => 4 iterations per row
    ROWS_PER_PROG = 64          # R=8192 => 128 blocks, good SM utilisation

    assert N % BLOCK_SIZE_CLASS == 0, "BLOCK_SIZE must divide N for mask‑free loop"

    num_blocks = triton.cdiv(R, ROWS_PER_PROG)
    partial_sums = torch.empty(num_blocks, dtype=torch.float32, device=logits.device)

    # Stage 1: per‑block loss accumulation
    cross_entropy_multirow_kernel[(num_blocks,)](
        logits, targets, partial_sums,
        N=N, BLOCK_SIZE=BLOCK_SIZE_CLASS,
        ROWS_PER_PROG=ROWS_PER_PROG, R=R,
        num_warps=8, num_stages=3,   # slightly more pipelining than parent
    )

    # Stage 2: final sum and division.
    # The reduction block must be large enough to load all partials.
    output = torch.empty(1, dtype=torch.float32, device=logits.device)
    reduce_mean_kernel[(1,)](
        partial_sums, output,
        R=R, NUM_PARTIALS=num_blocks,
        num_warps=4, num_stages=1,
    )

    return output.squeeze()