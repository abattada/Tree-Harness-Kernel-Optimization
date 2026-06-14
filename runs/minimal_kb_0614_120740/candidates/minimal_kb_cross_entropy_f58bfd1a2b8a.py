import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 1: per‑row online softmax + NLL loss, one program per row.
# Each row is processed in BLOCK_SIZE chunks.
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

    # Online softmax: compute max and sum of exp in one pass
    m_old = tl.full([], float('-inf'), dtype=tl.float32)
    d_old = tl.full([], 0.0, dtype=tl.float32)

    row_base = pid * N
    for start in range(0, N, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        # With BLOCK_SIZE dividing N exactly (N=32768, BLOCK_SIZE e.g. 1024/2048/4096), the mask is always true
        mask = offs < N
        x = tl.load(logits_ptr + row_base + offs, mask=mask, other=float('-inf'),
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
# Kernel 2a: first reduction stage – each block sums a chunk of per‑row losses
#            and writes a partial sum.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_sum_stage1_kernel(
    inp_ptr,          # f32 [R]
    out_partial_ptr,  # f32 [num_partials]
    R: tl.constexpr,  # total rows
    BLOCK_SIZE: tl.constexpr,  # rows per program
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
# Kernel 2b: second reduction stage – sums all partials and computes the mean.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_mean_stage2_kernel(
    inp_ptr,          # f32 [num_partials]
    out_scalar_ptr,   # f32 [1]
    R: tl.constexpr,           # total rows (for division)
    num_partials: tl.constexpr,  # number of partial sums
    BLOCK_SIZE: tl.constexpr,    # block size for this reduction (>= num_partials)
):
    total = tl.zeros([], dtype=tl.float32)
    for start in range(0, num_partials, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < num_partials
        vals = tl.load(inp_ptr + offs, mask=mask, other=0.0,
                       eviction_policy='evict_first')
        total += tl.sum(vals, axis=0)
    mean = total / R
    tl.store(out_scalar_ptr, mean, eviction_policy='evict_last')


# ---------------------------------------------------------------------------
# Main entry point: allocate outputs and launch kernels.
# ---------------------------------------------------------------------------
def triton_run(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    R, N = logits.shape
    device = logits.device

    # Params for the per‑row kernel: BLOCK_SIZE should divide N evenly.
    BLOCK_SIZE = 4096  # 32768 / 4096 = 8 iterations, good occupancy
    # Params for stage 1 reduction: number of rows per block
    STAGE1_BLOCK = 1024  # 8192 / 1024 = 8 partials
    # Params for stage 2 reduction: single block is enough
    STAGE2_BLOCK = 32

    # Allocate per‑row loss array
    loss_row = torch.empty(R, dtype=torch.float32, device=device)

    # Launch per‑row kernel (R programs, one per row)
    grid_per_row = (R,)
    cross_entropy_row_kernel[grid_per_row](
        logits, targets, loss_row,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
    )

    # Stage 1 reduction: sum groups of rows
    num_partials = (R + STAGE1_BLOCK - 1) // STAGE1_BLOCK
    partials = torch.empty(num_partials, dtype=torch.float32, device=device)
    grid_stage1 = (num_partials,)
    reduce_sum_stage1_kernel[grid_stage1](
        loss_row, partials,
        R=R,
        BLOCK_SIZE=STAGE1_BLOCK,
        num_warps=4,
        num_stages=2,
    )

    # Stage 2 reduction: sum partials and compute mean
    result = torch.empty(1, dtype=torch.float32, device=device)
    grid_stage2 = (1,)
    reduce_mean_stage2_kernel[grid_stage2](
        partials, result,
        R=R,
        num_partials=num_partials,
        BLOCK_SIZE=STAGE2_BLOCK,
        num_warps=4,
        num_stages=2,
    )

    return result