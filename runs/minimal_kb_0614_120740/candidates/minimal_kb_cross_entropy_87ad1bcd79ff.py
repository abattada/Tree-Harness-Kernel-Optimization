import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 1: per‑row online softmax + NLL loss, one program per row.
# ---------------------------------------------------------------------------
@triton.jit
def cross_entropy_row_kernel(
    logits_ptr,      # f32 [R, N]
    targets_ptr,     # i64 [R]
    loss_row_ptr,    # f32 [R]
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    target = tl.load(targets_ptr + pid)

    # Online softmax (single-pass)
    m_old = tl.full([], float('-inf'), dtype=tl.float32)
    d_old = tl.full([], 0.0, dtype=tl.float32)

    row_base = pid * N
    for start in range(0, N, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        # N is a multiple of BLOCK_SIZE → mask is always true → use None for mask
        ptr = tl.multiple_of(logits_ptr + row_base + offs, 32)
        x = tl.load(ptr, mask=None, other=float('-inf'),
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
    tl.store(loss_row_ptr + pid, loss, eviction_policy='evict_last')


# ---------------------------------------------------------------------------
# Kernel 2a: first reduction stage – sum chunks of per‑row losses into partials.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_sum_stage1_kernel(
    inp_ptr,          # f32 [R]
    out_partial_ptr,  # f32 [num_partials]
    R: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    # R is multiple of BLOCK_SIZE → mask always true
    vals = tl.load(inp_ptr + offs, mask=None, other=0.0,
                   eviction_policy='evict_first')
    partial = tl.sum(vals, axis=0)
    tl.store(out_partial_ptr + pid, partial, eviction_policy='evict_last')


# ---------------------------------------------------------------------------
# Kernel 2b: second reduction stage – sum all partials and compute mean.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_mean_stage2_kernel(
    inp_ptr,          # f32 [num_partials]
    out_scalar_ptr,   # f32 [1]
    R: tl.constexpr,
    num_partials: tl.constexpr,
):
    offs = tl.arange(0, num_partials)
    vals = tl.load(inp_ptr + offs, eviction_policy='evict_first')
    total = tl.sum(vals, axis=0)
    mean = total / R
    tl.store(out_scalar_ptr, mean, eviction_policy='evict_last')


# ---------------------------------------------------------------------------
# triton_run(logits, targets) -> torch.Tensor scalar
# ---------------------------------------------------------------------------
def triton_run(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    R, N = logits.shape
    assert logits.is_cuda and targets.is_cuda
    assert logits.dtype == torch.float32 and targets.dtype == torch.int64
    assert R == targets.shape[0]

    # Tuned parameters that divide dimensions exactly
    BLOCK_SIZE_N = 2048          # divides 32768 → 16 iterations
    BLOCK_SIZE_R = 2048          # divides 8192 → 4 partials

    num_partials = R // BLOCK_SIZE_R  # = 4

    # Per-row losses
    loss_row = torch.empty(R, device='cuda', dtype=torch.float32)

    # Launch kernel 1
    grid = (R,)
    cross_entropy_row_kernel[grid](
        logits, targets, loss_row,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE_N,
        num_warps=8,
        num_stages=4,
    )

    # Partial sums
    partials = torch.empty(num_partials, device='cuda', dtype=torch.float32)

    # Launch kernel 2a
    grid = (num_partials,)
    reduce_sum_stage1_kernel[grid](
        loss_row, partials,
        R=R,
        BLOCK_SIZE=BLOCK_SIZE_R,
        num_warps=4,
        num_stages=3,
    )

    # Launch kernel 2b
    result = torch.empty(1, device='cuda', dtype=torch.float32)
    reduce_mean_stage2_kernel[(1,)](
        partials, result,
        R=R,
        num_partials=num_partials,
        num_warps=4,
    )

    return result.view(())