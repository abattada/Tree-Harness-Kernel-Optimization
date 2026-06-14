import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 1: per‑row online softmax + nll loss, one program per row.
# Each row is processed in BLOCK_SIZE chunks. No masks needed because N is
# a multiple of BLOCK_SIZE.
# ---------------------------------------------------------------------------
@triton.jit
def cross_entropy_row_kernel(
    logits_ptr,      # f32 [R, N]
    targets_ptr,     # i64 [R]
    loss_row_ptr,    # f32 [R]   per‑row loss, to be reduced later
    N: tl.constexpr,           # number of classes
    BLOCK_SIZE: tl.constexpr,  # block along N dimension (must divide N)
):
    pid = tl.program_id(0)
    target = tl.load(targets_ptr + pid)

    # Online softmax: compute max and sum of exp in one pass
    m_old = tl.full([], float('-inf'), dtype=tl.float32)
    d_old = tl.full([], 0.0, dtype=tl.float32)

    row_base = pid * N
    for start in range(0, N, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        x = tl.load(logits_ptr + row_base + offs)

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
    tl.store(loss_row_ptr + pid, loss)


# ---------------------------------------------------------------------------
# Kernel 2a: first reduction stage – each block sums a chunk of per‑row losses
#            and writes a partial sum. No masks because R is a multiple of
#            BLOCK_SIZE.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_sum_stage1_kernel(
    inp_ptr,          # f32 [R]
    out_partial_ptr,  # f32 [num_partials]
    R: tl.constexpr,  # total rows
    BLOCK_SIZE: tl.constexpr,  # rows per program (must divide R)
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    vals = tl.load(inp_ptr + offs)
    partial = tl.sum(vals, axis=0)
    tl.store(out_partial_ptr + pid, partial)


# ---------------------------------------------------------------------------
# Kernel 2b: second reduction stage – sums all partials and computes the mean.
# No masks needed because num_partials is 8 and BLOCK_SIZE=1024 covers it.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_mean_stage2_kernel(
    inp_ptr,          # f32 [num_partials]
    out_scalar_ptr,   # f32 [1]
    R: tl.constexpr,           # total rows (for division)
    num_partials: tl.constexpr,  # number of partial sums
    BLOCK_SIZE: tl.constexpr,    # block size for this reduction (must cover num_partials)
):
    total = tl.zeros([], dtype=tl.float32)
    for start in range(0, num_partials, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        vals = tl.load(inp_ptr + offs)
        total += tl.sum(vals)
    mean = total / R
    tl.store(out_scalar_ptr, mean)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def triton_run(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    R, N = logits.shape
    assert targets.shape == (R,)
    assert logits.is_cuda and targets.is_cuda
    # Ensure divisibility for mask‑free code
    assert N % 4096 == 0, "N must be divisible by block size"
    assert R % 1024 == 0, "R must be divisible by reduction block size"

    loss_row = torch.empty(R, dtype=torch.float32, device=logits.device)

    # -----------------------------------------------------------------------
    # Kernel 1: per‑row loss
    # -----------------------------------------------------------------------
    BLOCK_SIZE = 8192  # each thread handles 32 elements (8 warps → 256 threads)
    grid_row = (R,)
    cross_entropy_row_kernel[grid_row](
        logits, targets, loss_row,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=2,
    )

    # -----------------------------------------------------------------------
    # Reduction: two‑stage sum over rows, then mean
    # -----------------------------------------------------------------------
    # Stage 1: sum chunks of rows
    BLOCK_SIZE_RED1 = 1024  # divides R=8192 exactly → no mask
    num_partials = R // BLOCK_SIZE_RED1
    partials = torch.empty(num_partials, dtype=torch.float32, device=logits.device)
    grid_red1 = (num_partials,)
    reduce_sum_stage1_kernel[grid_red1](
        loss_row, partials,
        R=R,
        BLOCK_SIZE=BLOCK_SIZE_RED1,
        num_warps=4,
        num_stages=2,
    )

    # Stage 2: combine partials and divide
    scalar_out = torch.empty((), dtype=torch.float32, device=logits.device)
    BLOCK_SIZE_RED2 = 1024  # > 8, covers all partials in one load
    reduce_mean_stage2_kernel[(1,)](
        partials, scalar_out,
        R=R,
        num_partials=num_partials,
        BLOCK_SIZE=BLOCK_SIZE_RED2,
        num_warps=4,
        num_stages=2,
    )

    return scalar_out