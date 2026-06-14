import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 1: per‑row online softmax + nll loss, one program per row.
# Each row is processed in BLOCK_SIZE chunks.
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
    # Load target class for this row
    target = tl.load(targets_ptr + pid)

    # Online softmax: compute max and sum of exp in one pass
    m_old = tl.full([], -1e9, dtype=tl.float32)   # near -inf, preserve stability
    d_old = tl.full([], 0.0, dtype=tl.float32)

    row_base = pid * N
    for start in range(0, N, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        # BLOCK_SIZE divides N exactly, so mask is always True,
        # but we keep it for generality – compiler will optimize.
        x = tl.load(
            logits_ptr + row_base + offs,
            mask=offs < N,
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
    tl.store(out_partial_ptr + pid, partial)


# ---------------------------------------------------------------------------
# Kernel 2b: second reduction stage – sums all partials and computes the mean.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_mean_stage2_kernel(
    inp_ptr,          # f32 [num_partials]
    out_scalar_ptr,   # f32 [1]
    R: tl.constexpr,           # total rows (for division)
    num_partials: tl.constexpr,  # number of partial sums
    BLOCK_SIZE: tl.constexpr,    # block size for this reduction (usually >= num_partials)
):
    total = tl.zeros([], dtype=tl.float32)
    for start in range(0, num_partials, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < num_partials
        v = tl.load(inp_ptr + offs, mask=mask, other=0.0,
                    eviction_policy='evict_first')
        total += tl.sum(v, axis=0)
    mean = total / R
    tl.store(out_scalar_ptr, mean)


# ---------------------------------------------------------------------------
# Main driver function
# ---------------------------------------------------------------------------
def triton_run(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    R, N = logits.shape
    assert targets.shape == (R,)
    assert logits.is_cuda and targets.is_cuda

    # Allocate per‑row losses
    loss_row = torch.empty(R, dtype=torch.float32, device=logits.device)

    # Row kernel: one program per row
    # BLOCK_SIZE chosen to divide 32768 exactly; 2048 gives 16 tiles per row.
    BLOCK_SIZE = 2048
    grid_row = (R,)
    cross_entropy_row_kernel[grid_row](
        logits,
        targets,
        loss_row,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,      # moderate warps for this workload
    )

    # First reduction stage: sum groups of rows into partials
    # Use 1024 rows per partial for 8 partials total.
    BLOCK_STAGE1 = 1024
    num_partials = (R + BLOCK_STAGE1 - 1) // BLOCK_STAGE1
    partials = torch.empty(num_partials, dtype=torch.float32, device=logits.device)
    grid_stage1 = (num_partials,)
    reduce_sum_stage1_kernel[grid_stage1](
        loss_row,
        partials,
        R=R,
        BLOCK_SIZE=BLOCK_SIZE,  # note: this is rows per program, not N
        num_warps=4,
    )

    # Second reduction stage: sum partials and compute mean
    # Use a single program with BLOCK_SIZE >= num_partials
    BLOCK_STAGE2 = 1024
    out = torch.empty(1, dtype=torch.float32, device=logits.device)
    reduce_mean_stage2_kernel[(1,)](
        partials,
        out,
        R=R,
        num_partials=num_partials,
        BLOCK_SIZE=BLOCK_STAGE2,
        num_warps=4,
    )

    return out