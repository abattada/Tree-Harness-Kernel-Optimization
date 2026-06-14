import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 1: per‑row online softmax + nll loss, one program per row.
# Each row is processed in BLOCK_SIZE chunks along the class dimension.
# Because N is an exact multiple of BLOCK_SIZE, no boundary masking is needed.
# ---------------------------------------------------------------------------
@triton.jit
def cross_entropy_online_kernel(
    logits_ptr,      # f32 [R, N]
    targets_ptr,     # i64 [R]
    loss_row_ptr,    # f32 [R]   per‑row loss, to be reduced later
    N: tl.constexpr,           # number of classes
    BLOCK_SIZE: tl.constexpr,  # chunk size along class dimension
):
    tl.static_assert(N % BLOCK_SIZE == 0)  # ensure no partial final chunk

    pid = tl.program_id(0)
    target = tl.load(targets_ptr + pid)

    m_old = tl.full([], float('-inf'), dtype=tl.float32)
    d_old = tl.full([], 0.0, dtype=tl.float32)

    row_base = pid * N
    for start in range(0, N, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        # No mask needed – BLOCK_SIZE divides N exactly
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
    tl.store(loss_row_ptr + pid, loss, eviction_policy='evict_last')


# ---------------------------------------------------------------------------
# Kernel 2a: first reduction stage – each block sums a chunk of per‑row losses
#            and writes a partial sum.  BLOCK_SIZE_REDUCE divides R exactly.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_sum_stage1_kernel(
    inp_ptr,          # f32 [R]
    out_partial_ptr,  # f32 [num_partials]
    R: tl.constexpr,  # total rows
    BLOCK_SIZE: tl.constexpr,  # rows per program
):
    tl.static_assert(R % BLOCK_SIZE == 0)  # no partial block
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    vals = tl.load(inp_ptr + offs, eviction_policy='evict_first')
    partial = tl.sum(vals, axis=0)
    tl.store(out_partial_ptr + pid, partial)


# ---------------------------------------------------------------------------
# Kernel 2b: second reduction stage – sums all partials and computes the mean.
#            BLOCK_SIZE_FINAL is large enough to cover all partials at once.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_mean_stage2_kernel(
    inp_ptr,          # f32 [num_partials]
    out_scalar_ptr,   # f32 [1]
    R: tl.constexpr,           # total rows (for division)
    num_partials: tl.constexpr,  # number of partial sums
    BLOCK_SIZE: tl.constexpr,    # block size for this reduction
):
    total = tl.zeros([], dtype=tl.float32)
    for start in range(0, num_partials, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < num_partials
        vals = tl.load(
            inp_ptr + offs,
            mask=mask,
            other=0.0,
            eviction_policy='evict_first',
        )
        total += tl.sum(vals, axis=0)
    mean = total / R
    tl.store(out_scalar_ptr, mean, eviction_policy='evict_last')


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

    # Block sizes tuned for the given shape; powers of two that divide the
    # dimensions exactly and balance occupancy vs register pressure.
    BLOCK_SIZE_CLASS = 4096    # chunk along class dimension
    BLOCK_SIZE_REDUCE = 1024   # rows per reduction block
    BLOCK_SIZE_FINAL  = 1024   # block size for final summation (> num_partials)

    # Buffer for per-row losses
    loss_row = torch.empty(R, dtype=torch.float32, device=logits.device)

    # Stage 1: compute loss for every row
    grid_per_row = (R,)
    cross_entropy_online_kernel[grid_per_row](
        logits, targets, loss_row,
        N=N, BLOCK_SIZE=BLOCK_SIZE_CLASS,
        num_warps=8,            # improves latency hiding for memory-heavy loads
    )

    # Stage 2a: reduce rows into partial sums
    num_partials = triton.cdiv(R, BLOCK_SIZE_REDUCE)
    partials = torch.empty(num_partials, dtype=torch.float32, device=logits.device)
    grid_stage1 = (num_partials,)
    reduce_sum_stage1_kernel[grid_stage1](
        loss_row, partials,
        R=R, BLOCK_SIZE=BLOCK_SIZE_REDUCE,
        num_warps=4,
    )

    # Stage 2b: sum partials and divide by R to get mean
    output = torch.empty(1, dtype=torch.float32, device=logits.device)
    reduce_mean_stage2_kernel[(1,)](
        partials, output,
        R=R, num_partials=num_partials, BLOCK_SIZE=BLOCK_SIZE_FINAL,
        num_warps=4,
    )

    return output.squeeze()