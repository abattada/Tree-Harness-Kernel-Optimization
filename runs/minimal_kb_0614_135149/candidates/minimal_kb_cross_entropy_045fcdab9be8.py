import torch
import triton
import triton.language as tl

# --------------------------------------------------------------------------
# Kernel 1: per-row online softmax + NLL loss
# --------------------------------------------------------------------------
@triton.jit
def cross_entropy_row_kernel(
    logits_ptr,          # f32 [R, N]
    targets_ptr,         # i64 [R]
    loss_row_ptr,        # f32 [R]  per-row loss
    N: tl.constexpr,     # number of classes
    BLOCK_SIZE: tl.constexpr,  # chunk size along N
):
    pid = tl.program_id(0)          # row index
    target = tl.load(targets_ptr + pid)

    m_old = tl.full([], float('-inf'), dtype=tl.float32)
    d_old = tl.full([], 0.0, dtype=tl.float32)

    row_base = pid * N
    for start in range(0, N, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(
            logits_ptr + row_base + offs,
            mask=mask,
            other=float('-inf'),
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


# --------------------------------------------------------------------------
# Kernel 2a: first reduction stage – partial sums of per-row losses
# --------------------------------------------------------------------------
@triton.jit
def reduce_sum_stage1_kernel(
    inp_ptr,             # f32 [R]  per-row losses
    out_partial_ptr,     # f32 [num_partials]
    R: tl.constexpr,     # total number of rows
    BLOCK_SIZE: tl.constexpr,  # rows processed per program
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < R
    vals = tl.load(inp_ptr + offs, mask=mask, other=0.0,
                   eviction_policy='evict_first')
    partial = tl.sum(vals, axis=0)
    tl.store(out_partial_ptr + pid, partial)


# --------------------------------------------------------------------------
# Kernel 2b: second reduction – sums partials and computes the mean
# --------------------------------------------------------------------------
@triton.jit
def reduce_mean_stage2_kernel(
    inp_ptr,              # f32 [num_partials]
    out_ptr,              # f32 [1]  output scalar
    R: tl.constexpr,      # total rows, for division
    num_partials: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # block size to cover all partials
):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < num_partials
    vals = tl.load(inp_ptr + offs, mask=mask, other=0.0)
    total = tl.sum(vals, axis=0)
    mean = total / R
    if tl.program_id(0) == 0:
        tl.store(out_ptr, mean)


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def triton_run(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits:  f32[8192, 32768]
    targets: i64[8192]
    returns: f32 scalar (mean cross-entropy)
    """
    R, N = logits.shape
    assert targets.shape == (R,), "Expected 1D targets"
    device = logits.device

    # --- choose block sizes (sensible defaults, divisible by shape) ---
    BLOCK_SIZE_N = 1024           # 32768 / 1024 = 32 iterations per row
    BLOCK_SIZE_R = 256            # 8192  / 256  = 32 partial sums
    BLOCK_SIZE_S2 = 64            # > 32, easily covers all partials

    assert N % BLOCK_SIZE_N == 0
    assert R % BLOCK_SIZE_R == 0

    num_partials = R // BLOCK_SIZE_R

    # per-row losses
    loss_row = torch.empty(R, dtype=torch.float32, device=device)

    # launch row kernel (one program per row)
    cross_entropy_row_kernel[(R,)](
        logits, targets, loss_row,
        N, BLOCK_SIZE_N,
    )

    # stage 1 reduction
    partials = torch.empty(num_partials, dtype=torch.float32, device=device)
    reduce_sum_stage1_kernel[(num_partials,)](
        loss_row, partials,
        R, BLOCK_SIZE_R,
    )

    # stage 2 reduction -> scalar
    out = torch.empty(1, dtype=torch.float32, device=device)
    reduce_mean_stage2_kernel[(1,)](
        partials, out,
        R, num_partials, BLOCK_SIZE_S2,
    )

    return out.view(())  # return 0-dim tensor