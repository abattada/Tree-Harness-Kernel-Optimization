import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 1: per‑row online softmax + nll loss, one program per row.
# Chunk size along N is chosen so that N is a perfect multiple -> masks dead.
# ---------------------------------------------------------------------------
@triton.jit
def cross_entropy_row_kernel(
    logits_ptr,        # f32 [R, N]
    targets_ptr,       # i64 [R]
    loss_row_ptr,      # f32 [R]   per‑row loss, reduced later
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    target = tl.load(targets_ptr + pid)

    m = tl.full([], float('-inf'), dtype=tl.float32)
    d = tl.full([], 0.0, dtype=tl.float32)

    row_base = pid * N
    for start in range(0, N, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        # N is divisible by BLOCK_SIZE; no mask nor 'other' needed.
        x = tl.load(
            logits_ptr + row_base + offs,
            eviction_policy='evict_first',
        )
        m_loc = tl.max(x, axis=0)
        m_new = tl.maximum(m, m_loc)
        exp_centered = tl.exp(x - m_new)
        sum_exp = tl.sum(exp_centered, axis=0)
        d = d * tl.exp(m - m_new) + sum_exp
        m = m_new

    logsumexp = m + tl.log(d)

    # Load the logit of the target class and compute the row loss.
    target_logit = tl.load(logits_ptr + row_base + target)
    loss = logsumexp - target_logit
    tl.store(loss_row_ptr + pid, loss, eviction_policy='evict_last')


# ---------------------------------------------------------------------------
# Kernel 2a: first reduction stage – each block sums a chunk of per‑row losses
#            and writes a partial sum.  R is divisible by BLOCK_SIZE.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_sum_stage1_kernel(
    inp_ptr,            # f32 [R]
    out_partial_ptr,    # f32 [num_partials]
    R: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # rows per program, divides R
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    # No mask needed when R % BLOCK_SIZE == 0.
    vals = tl.load(inp_ptr + offs, eviction_policy='evict_first')
    partial = tl.sum(vals, axis=0)
    tl.store(out_partial_ptr + pid, partial)


# ---------------------------------------------------------------------------
# Kernel 2b: second reduction stage – sums all partials and computes the mean.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_mean_stage2_kernel(
    inp_ptr,            # f32 [num_partials]
    out_scalar_ptr,     # f32 [1]
    R: tl.constexpr,    # total rows (for division)
    num_partials: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
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
# Main entry point
# ---------------------------------------------------------------------------
def triton_run(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute mean cross-entropy loss.

    logits:  float32 [8192, 32768]
    targets: int64   [8192]
    returns: scalar float32 loss
    """
    R, N = logits.shape
    assert logits.dtype == torch.float32 and targets.dtype == torch.int64

    # ------------ tunable knobs ------------
    # N = 32768 -> divisible by 8192, 4096, 2048, …
    BLOCK_SIZE_CLASS = 8192          # 4 iterations per row, no masks
    BLOCK_SIZE_REDUCE = 1024         # divides R=8192 (8 partials)
    BLOCK_SIZE_FINAL = 1024          # covers all partials with a mask
    # ---------------------------------------

    loss_row = torch.empty(R, dtype=torch.float32, device=logits.device)

    # Stage 1: row‑wise loss
    grid_rows = (R,)
    cross_entropy_row_kernel[grid_rows](
        logits, targets, loss_row,
        N=N, BLOCK_SIZE=BLOCK_SIZE_CLASS,
        num_warps=8, num_stages=4,       # pipeline loads + compute
    )

    # Stage 2a: sum rows → partials
    num_partials = triton.cdiv(R, BLOCK_SIZE_REDUCE)
    partials = torch.empty(num_partials, dtype=torch.float32, device=logits.device)
    grid_s1 = (num_partials,)
    reduce_sum_stage1_kernel[grid_s1](
        loss_row, partials,
        R=R, BLOCK_SIZE=BLOCK_SIZE_REDUCE,
    )

    # Stage 2b: sum partials → mean
    output = torch.empty(1, dtype=torch.float32, device=logits.device)
    grid_s2 = (1,)
    reduce_mean_stage2_kernel[grid_s2](
        partials, output,
        R=R, num_partials=num_partials, BLOCK_SIZE=BLOCK_SIZE_FINAL,
    )

    return output.squeeze()