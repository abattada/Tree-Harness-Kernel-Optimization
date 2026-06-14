import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Fused kernel: per‑row online softmax + NLL loss, accumulated across multiple
# rows per program.  Writes out a partial sum per program for the final mean.
# ---------------------------------------------------------------------------
@triton.jit
def cross_entropy_fused_kernel(
    logits_ptr,     # f32 [R, N]
    targets_ptr,    # i64 [R]
    partials_ptr,   # f32 [num_partials]  (per‑program sum of row losses)
    R: tl.constexpr,              # total number of rows
    N: tl.constexpr,              # number of classes
    ROWS_PER_PROG: tl.constexpr,  # rows handled by each program
    BLOCK_SIZE: tl.constexpr,     # block along the N dimension
):
    pid = tl.program_id(0)
    start_row = pid * ROWS_PER_PROG
    end_row = start_row + ROWS_PER_PROG
    if end_row > R:
        end_row = R

    acc = tl.zeros([], dtype=tl.float32)

    for row_idx in range(start_row, end_row):
        target = tl.load(targets_ptr + row_idx)

        # Online softmax (single pass)
        m_old = tl.full([], float('-inf'), dtype=tl.float32)
        d_old = tl.full([], 0.0, dtype=tl.float32)
        row_base = row_idx * N

        for start in range(0, N, BLOCK_SIZE):
            offs = start + tl.arange(0, BLOCK_SIZE)
            mask = offs < N
            x = tl.load(logits_ptr + row_base + offs, mask=mask, other=float('-inf'))

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

    tl.store(partials_ptr + pid, acc)


# ---------------------------------------------------------------------------
# Final reduction kernel: sums all partials and divides by R -> mean.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_mean_kernel(
    partials_ptr,     # f32 [num_partials]
    out_scalar_ptr,   # f32 [1]
    R: tl.constexpr,               # total rows
    num_partials: tl.constexpr,    # number of partial sums
    BLOCK_SIZE: tl.constexpr,      # block size for sequential sum
):
    total = tl.zeros([], dtype=tl.float32)
    for start in range(0, num_partials, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < num_partials
        vals = tl.load(partials_ptr + offs, mask=mask, other=0.0)
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

    ROWS_PER_PROG = 64          # → 128 programs
    num_partials = (R + ROWS_PER_PROG - 1) // ROWS_PER_PROG
    partials = torch.empty(num_partials, dtype=torch.float32, device=logits.device)

    # Fused row loss + first reduction
    grid = (num_partials,)
    BLOCK_SIZE = 4096
    cross_entropy_fused_kernel[grid](
        logits, targets, partials,
        R, N,
        ROWS_PER_PROG=ROWS_PER_PROG,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=2,
    )

    # Final reduction
    scalar_out = torch.empty((), dtype=torch.float32, device=logits.device)
    reduce_mean_kernel[(1,)](
        partials, scalar_out,
        R, num_partials,
        BLOCK_SIZE=1024,
        num_warps=4,
        num_stages=2,
    )

    return scalar_out