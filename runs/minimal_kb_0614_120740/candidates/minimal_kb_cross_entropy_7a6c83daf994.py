import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 1: per‑row online softmax + nll loss, one program per row.
# ---------------------------------------------------------------------------
@triton.jit
def cross_entropy_row_kernel(
    logits_ptr,      # f32 [R, N]
    targets_ptr,     # i64 [R]
    loss_row_ptr,    # f32 [R]   per‑row loss, to be reduced later
    N: tl.constexpr,           # number of classes
    BLOCK_SIZE: tl.constexpr,  # block along N dimension
):
    pid = tl.program_id(0)
    # Load target class for this row
    target = tl.load(targets_ptr + pid)

    # Online softmax: compute logsumexp in one pass
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

    # Load target logit
    target_logit = tl.load(logits_ptr + row_base + target)
    loss = logsumexp - target_logit
    tl.store(loss_row_ptr + pid, loss, eviction_policy='evict_last')


# ---------------------------------------------------------------------------
# Kernel 2a: first reduction stage – sum per‑row losses in groups
# ---------------------------------------------------------------------------
@triton.jit
def reduce_sum_stage1_kernel(
    inp_ptr,          # f32 [R]
    out_partial_ptr,  # f32 [num_partials]
    R: tl.constexpr,  # total rows
    BLOCK_SIZE: tl.constexpr,  # group size per program
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < R
    vals = tl.load(inp_ptr + offs, mask=mask, other=0.0,
                   eviction_policy='evict_first')
    partial = tl.sum(vals, axis=0)
    tl.store(out_partial_ptr + pid, partial, eviction_policy='evict_last')


# ---------------------------------------------------------------------------
# Kernel 2b: second reduction stage – sum partials and compute mean
# ---------------------------------------------------------------------------
@triton.jit
def reduce_mean_stage2_kernel(
    inp_ptr,          # f32 [num_partials]
    out_scalar_ptr,   # f32 [1]
    R: tl.constexpr,           # total rows (for division)
    num_partials: tl.constexpr,  # number of partials
    BLOCK_SIZE: tl.constexpr,    # block size (>= num_partials)
):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < num_partials
    vals = tl.load(inp_ptr + offs, mask=mask, other=0.0,
                   eviction_policy='evict_first')
    total = tl.sum(vals, axis=0)
    mean = total / R
    tl.store(out_scalar_ptr, mean)


# ---------------------------------------------------------------------------
# Top‑level driver
# ---------------------------------------------------------------------------
def triton_run(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    R, N = logits.shape
    assert logits.dtype == torch.float32
    assert targets.dtype == torch.int64
    assert targets.shape[0] == R

    # Allocate per‑row loss buffer
    loss_row = torch.empty(R, device='cuda', dtype=torch.float32)

    # Launch per‑row kernel
    BLOCK_SIZE = 1024  # divides N=32768 exactly
    grid = (R,)
    cross_entropy_row_kernel[grid](
        logits, targets, loss_row,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
    )

    # Two‑stage reduction to scalar mean
    BLOCK_SIZE_STAGE1 = 1024  # 8192/1024 = 8 partials
    num_partials = triton.cdiv(R, BLOCK_SIZE_STAGE1)
    partials = torch.empty(num_partials, device='cuda', dtype=torch.float32)

    reduce_sum_stage1_kernel[(num_partials,)](
        loss_row, partials,
        R=R,
        BLOCK_SIZE=BLOCK_SIZE_STAGE1,
        num_warps=4,
        num_stages=2,
    )

    # Second stage: sum partials and divide by R
    out = torch.empty(1, device='cuda', dtype=torch.float32)
    BLOCK_SIZE_STAGE2 = triton.next_power_of_2(num_partials)
    reduce_mean_stage2_kernel[(1,)](
        partials, out,
        R=R,
        num_partials=num_partials,
        BLOCK_SIZE=BLOCK_SIZE_STAGE2,
        num_warps=4,
        num_stages=2,
    )

    return out.squeeze()