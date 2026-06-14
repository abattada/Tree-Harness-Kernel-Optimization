import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: per‑row online softmax + NLL loss, one program per row.
# Each row is processed in BLOCK_SIZE_N chunks along the class dimension.
# Masks are omitted because BLOCK_SIZE_N divides N exactly.
# ---------------------------------------------------------------------------
@triton.jit
def cross_entropy_row_kernel(
    logits_ptr,       # f32 [R, N]
    targets_ptr,      # i64 [R]
    loss_row_ptr,     # f32 [R]
    N: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    tl.static_assert(N % BLOCK_SIZE_N == 0, "N must be a multiple of BLOCK_SIZE_N")
    pid = tl.program_id(0)
    target = tl.load(targets_ptr + pid)

    # Online softmax: compute max and log-sum-exp in a single pass
    m_curr = tl.full([], float('-inf'), dtype=tl.float32)
    d_curr = tl.full([], 0.0, dtype=tl.float32)

    row_base = pid * N
    for start in range(0, N, BLOCK_SIZE_N):
        offs = start + tl.arange(0, BLOCK_SIZE_N)
        x = tl.load(
            logits_ptr + row_base + offs,
            eviction_policy='evict_first',
        )

        m_loc = tl.max(x, axis=0)
        m_new = tl.maximum(m_curr, m_loc)
        exp_centered = tl.exp(x - m_new)
        sum_exp = tl.sum(exp_centered, axis=0)
        d_new = d_curr * tl.exp(m_curr - m_new) + sum_exp
        m_curr = m_new
        d_curr = d_new

    # logsumexp = max + log(sum(exp(x - max)))
    logsumexp = m_curr + tl.log(d_curr)

    # Load the logit of the correct class and compute loss
    target_logit = tl.load(logits_ptr + row_base + target)
    loss = logsumexp - target_logit
    tl.store(loss_row_ptr + pid, loss, eviction_policy='evict_last')


# ---------------------------------------------------------------------------
# Kernel 2a: first reduction stage – each block sums a chunk of per‑row losses
#            and writes a partial sum. Mask omitted because BLOCK_SIZE_R
#            divides R exactly.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_sum_stage1_kernel(
    inp_ptr,          # f32 [R]
    out_partial_ptr,  # f32 [num_partials]
    R: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
):
    tl.static_assert(R % BLOCK_SIZE_R == 0, "R must be a multiple of BLOCK_SIZE_R")
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE_R
    offs = start + tl.arange(0, BLOCK_SIZE_R)
    vals = tl.load(inp_ptr + offs, eviction_policy='evict_first')
    partial = tl.sum(vals, axis=0)
    tl.store(out_partial_ptr + pid, partial)


# ---------------------------------------------------------------------------
# Kernel 2b: second reduction stage – sums all partials and computes the mean.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_mean_stage2_kernel(
    inp_ptr,          # f32 [num_partials]
    out_scalar_ptr,   # f32 [] – scalar output
    R: tl.constexpr,
    num_partials: tl.constexpr,
    BLOCK_SIZE_R2: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE_R2)
    mask = offs < num_partials
    partials = tl.load(inp_ptr + offs, mask=mask, other=0.0)
    total = tl.sum(partials, axis=0)
    mean = total / tl.full([], R, dtype=tl.float32)
    tl.store(out_scalar_ptr, mean)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def triton_run(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits : float32 [R, N]  (R = 8192, N = 32768)
    targets: int64   [R]
    returns: float32 scalar = mean cross-entropy loss
    """
    R, N = logits.shape
    assert targets.shape == (R,)
    assert logits.dtype == torch.float32
    assert targets.dtype == torch.int64

    # Tuned constants
    BLOCK_SIZE_N = 4096   # divides 32768
    BLOCK_SIZE_R = 512    # divides 8192
    BLOCK_SIZE_R2 = 1024  # >= num_partials (16)
    NUM_WARPS_ROW = 8     # 4096 elems → 16 per thread
    NUM_WARPS_R1  = 4     # 512 elems → 4 per thread
    NUM_WARPS_R2  = 1     # tiny final reduction

    # Per-row losses
    loss_row = torch.empty(R, dtype=torch.float32, device=logits.device)

    grid_rows = (R,)
    cross_entropy_row_kernel[grid_rows](
        logits, targets, loss_row,
        N, BLOCK_SIZE_N,
        num_warps=NUM_WARPS_ROW,
    )

    # First-stage reduction: partial sums
    num_partials = R // BLOCK_SIZE_R
    partials = torch.empty(num_partials, dtype=torch.float32, device=logits.device)
    reduce_sum_stage1_kernel[(num_partials,)](
        loss_row, partials,
        R, BLOCK_SIZE_R,
        num_warps=NUM_WARPS_R1,
    )

    # Second-stage reduction: sum partials → scalar mean
    out = torch.empty((), dtype=torch.float32, device=logits.device)
    reduce_mean_stage2_kernel[(1,)](
        partials, out,
        R, num_partials, BLOCK_SIZE_R2,
        num_warps=NUM_WARPS_R2,
    )

    return out