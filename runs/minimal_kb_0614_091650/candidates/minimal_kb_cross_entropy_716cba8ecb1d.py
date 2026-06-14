import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Fused kernel: multiple rows per program, compute per-row online softmax
# and accumulate the loss, then write a partial sum.
# ---------------------------------------------------------------------------
@triton.jit
def cross_entropy_fused_kernel(
    logits_ptr,        # f32 [R, N]
    targets_ptr,       # i64 [R]
    partials_ptr,      # f32 [num_programs]   output partial sums
    N: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,   # tile along N
):
    pid = tl.program_id(0)
    start_row = pid * ROWS_PER_PROG
    acc = tl.full([], 0.0, dtype=tl.float32)

    for row_idx in range(ROWS_PER_PROG):
        row = start_row + row_idx
        target = tl.load(targets_ptr + row)

        # Online softmax: one pass, compute logsumexp
        m_old = tl.full([], float('-inf'), dtype=tl.float32)
        d_old = tl.full([], 0.0, dtype=tl.float32)

        for start in range(0, N, BLOCK_SIZE):
            offs = start + tl.arange(0, BLOCK_SIZE)
            mask = offs < N
            x = tl.load(logits_ptr + row * N + offs, mask=mask, other=float('-inf'))

            m_loc = tl.max(x, axis=0)
            m_new = tl.maximum(m_old, m_loc)
            exp_centered = tl.exp(x - m_new)
            sum_exp = tl.sum(exp_centered, axis=0)
            d_new = d_old * tl.exp(m_old - m_new) + sum_exp
            m_old = m_new
            d_old = d_new

        logsumexp = m_old + tl.log(d_old)
        target_logit = tl.load(logits_ptr + row * N + target)
        loss = logsumexp - target_logit
        acc += loss

    tl.store(partials_ptr + pid, acc)


# ---------------------------------------------------------------------------
# Final reduction: sum all partial sums and divide by number of rows.
# Only one program is launched.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_final_kernel(
    partials_ptr,         # f32 [num_programs]
    scalar_ptr,           # f32 [1]
    R: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,   # >= NUM_PROGRAMS
):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < NUM_PROGRAMS
    parts = tl.load(partials_ptr + offs, mask=mask, other=0.0)
    total = tl.sum(parts)
    mean = total / R
    if tl.program_id(0) == 0:
        tl.store(scalar_ptr, mean)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def triton_run(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    R, N = logits.shape
    assert targets.shape == (R,)
    assert logits.is_cuda and targets.is_cuda
    assert logits.dtype == torch.float32 and targets.dtype == torch.int64

    # Tuned launch parameters
    NUM_PROGRAMS = 64           # fuses 8192/64 = 128 rows per program
    ROWS_PER_PROG = R // NUM_PROGRAMS
    assert R % NUM_PROGRAMS == 0, "R must be divisible by NUM_PROGRAMS"
    BLOCK_SIZE = 1024           # divides N=32768, good for coalescing
    BLOCK_SIZE_RED = 64         # >= NUM_PROGRAMS

    # Allocate output buffers
    partials = torch.empty(NUM_PROGRAMS, dtype=torch.float32, device=logits.device)
    scalar_out = torch.empty(1, dtype=torch.float32, device=logits.device)

    # Launch fused row + partial reduction kernel
    grid = (NUM_PROGRAMS,)
    cross_entropy_fused_kernel[grid](
        logits, targets, partials,
        N, ROWS_PER_PROG, BLOCK_SIZE,
        num_warps=8,
    )

    # Launch final reduction kernel (single program)
    grid_red = (1,)
    reduce_final_kernel[grid_red](
        partials, scalar_out,
        R, NUM_PROGRAMS, BLOCK_SIZE_RED,
        num_warps=4,
    )

    return scalar_out.squeeze(0)