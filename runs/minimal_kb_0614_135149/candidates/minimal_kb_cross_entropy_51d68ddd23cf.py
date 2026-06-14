import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 1: fused per-row online softmax + NLL loss + partial reduction.
# Each program handles several rows (ROWS_PER_PROG), accumulates their loss,
# and writes one partial sum.
# ---------------------------------------------------------------------------
@triton.jit
def cross_entropy_partial_kernel(
    logits_ptr,               # f32 [R, N]
    targets_ptr,              # i64 [R]
    partials_out_ptr,         # f32 [num_partials]
    R: tl.constexpr,          # total rows
    N: tl.constexpr,          # number of classes
    BLOCK_SIZE: tl.constexpr, # chunk size along class dimension
    ROWS_PER_PROG: tl.constexpr, # rows per program
):
    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)

    # Determine the range of rows this program is responsible for.
    first_row = pid * ROWS_PER_PROG
    last_row = first_row + ROWS_PER_PROG
    if last_row > R:
        last_row = R

    acc = tl.zeros([], dtype=tl.float32)

    for row_idx in range(first_row, last_row):
        target = tl.load(targets_ptr + row_idx)
        m_old = tl.full([], float('-inf'), dtype=tl.float32)
        d_old = tl.full([], 0.0, dtype=tl.float32)

        row_base = row_idx * N
        # Since 32768 % 4096 == 0, we can drop the mask entirely to save a
        # few instructions.  The static assert tells the compiler the mask is
        # always active, allowing full vectorization.
        if N % BLOCK_SIZE == 0 and BLOCK_SIZE % 128 == 0:
            # No mask needed
            for start in range(0, N, BLOCK_SIZE):
                offs = start + tl.arange(0, BLOCK_SIZE)
                x = tl.load(
                    logits_ptr + row_base + offs,
                    mask=None,
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
        else:
            # Fallback with mask for odd sizes (not used here, but kept for safety).
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
        acc += loss

    tl.store(partials_out_ptr + pid, acc, eviction_policy='evict_last')


# ---------------------------------------------------------------------------
# Kernel 2: sum partials and compute mean in a single block.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_mean_kernel(
    inp_ptr,              # f32 [num_partials]
    out_scalar_ptr,       # f32 [1]
    R: tl.constexpr,      # total rows (for division)
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
    mean = total / R.to(tl.float32)
    tl.store(out_scalar_ptr, mean, eviction_policy='evict_last')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def triton_run(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    R, N = logits.shape
    assert logits.dtype == torch.float32 and targets.dtype == torch.int64

    # Constants chosen for this specific shape (8192, 32768).
    BLOCK_SIZE = 4096          # class-dimension tile; divides N exactly
    ROWS_PER_PROG = 32         # rows handled by one program
    BLOCK_REDUCE = 1024        # block size for final reduction (covers all partials)

    # Number of partial sums = ceil(R / ROWS_PER_PROG)
    num_partials = triton.cdiv(R, ROWS_PER_PROG)
    partials = torch.empty(num_partials, dtype=torch.float32, device=logits.device)

    # Launch fused compute + partial reduction
    grid = (num_partials,)
    cross_entropy_partial_kernel[grid](
        logits, targets, partials,
        R=R, N=N,
        BLOCK_SIZE=BLOCK_SIZE,
        ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=4,      # each program has enough ILP
        num_stages=2,
    )

    # Final single-block reduction + mean
    output = torch.empty(1, dtype=torch.float32, device=logits.device)
    reduce_mean_kernel[(1,)](
        partials, output,
        R=R, num_partials=num_partials, BLOCK_SIZE=BLOCK_REDUCE,
        num_warps=4,
    )

    return output.squeeze()