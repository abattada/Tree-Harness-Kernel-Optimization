import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 1: per‑row online softmax + cross-entropy loss.
# Each program processes one row of shape [N].
# ---------------------------------------------------------------------------
@triton.jit
def cross_entropy_row_kernel(
    logits_ptr,   # f32 [R, N]
    targets_ptr,  # i64 [R]
    loss_row_ptr, # f32 [R]   (output per‑row loss)
    N: tl.constexpr,             # number of classes
    BLOCK_SIZE: tl.constexpr     # tile along the N dimension
):
    pid = tl.program_id(0)                     # row index
    target = tl.load(targets_ptr + pid)        # scalar i64

    # Online softmax: compute max and sum of exp in a single pass
    m_old = tl.full([], float('-inf'), dtype=tl.float32)
    d_old = tl.full([], 0.0, dtype=tl.float32)

    for start in range(0, N, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        # load a block of logits (all finite, but keep mask for generality)
        x = tl.load(logits_ptr + pid * N + offs, mask=mask, other=float('-inf'))

        # local max and sum of exponentials after centering with m_new
        m_loc = tl.max(x, axis=0)
        m_new = tl.maximum(m_old, m_loc)
        exp_centered = tl.exp(x - m_new)
        sum_exp = tl.sum(exp_centered, axis=0)

        # online update: scale previous contributions and add new
        d_new = d_old * tl.exp(m_old - m_new) + sum_exp
        m_old = m_new
        d_old = d_new

    # logsumexp = m_old + log(d_old)
    logsumexp = m_old + tl.log(d_old)

    # load the logit at the target position
    target_logit = tl.load(logits_ptr + pid * N + target)

    # per‑row loss = logsumexp - logit[target]
    loss = logsumexp - target_logit
    tl.store(loss_row_ptr + pid, loss)


# ---------------------------------------------------------------------------
# Kernel 2a: first reduction stage – sum a contiguous chunk of per‑row losses.
# Grid size = GRID (each program sums ROWS_PER_PROG rows).
# ---------------------------------------------------------------------------
@triton.jit
def reduce_stage1_kernel(
    inp_ptr,          # f32 [R]
    out_partial_ptr,  # f32 [GRID]
    ROWS_PER_PROG: tl.constexpr,   # number of rows per program (constant)
    BLOCK_SIZE_RED1: tl.constexpr   # load tile (≥ ROWS_PER_PROG)
):
    pid = tl.program_id(0)
    start = pid * ROWS_PER_PROG
    offs = start + tl.arange(0, BLOCK_SIZE_RED1)
    # all rows exist, so mask is always true (ROWS_PER_PROG ≤ BLOCK_SIZE_RED1)
    vals = tl.load(inp_ptr + offs, mask=offs < start + ROWS_PER_PROG, other=0.0)
    partial = tl.sum(vals, axis=0)
    tl.store(out_partial_ptr + pid, partial)


# ---------------------------------------------------------------------------
# Kernel 2b: second reduction stage – sum all partial sums and divide by R.
# Only one program is launched.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_stage2_kernel(
    partial_ptr,         # f32 [GRID]
    scalar_ptr,          # f32 [1]
    R: tl.constexpr,               # total number of rows
    GRID: tl.constexpr,            # number of partials
    BLOCK_SIZE_RED2: tl.constexpr  # tile (≥ GRID)
):
    offs = tl.arange(0, BLOCK_SIZE_RED2)
    mask = offs < GRID
    parts = tl.load(partial_ptr + offs, mask=mask, other=0.0)
    total = tl.sum(parts)
    mean = total / R
    # all threads have the same value; store by one thread
    if tl.program_id(0) == 0:
        tl.store(scalar_ptr, mean)


# ---------------------------------------------------------------------------
# Main entry point: allocates outputs, launches kernels, returns scalar.
# ---------------------------------------------------------------------------
def triton_run(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    R, N = logits.shape
    assert targets.shape == (R,)
    assert logits.is_cuda and targets.is_cuda
    assert logits.dtype == torch.float32 and targets.dtype == torch.int64

    # Launch configuration
    BLOCK_SIZE = 1024          # tile along N (divides 32768)
    GRID = 32                  # number of programs for stage1 reduction
    ROWS_PER_PROG = R // GRID  # must be integer (8192 / 32 = 256)
    assert R % GRID == 0, "R must be divisible by GRID"

    # Allocate intermediate and output buffers
    loss_row = torch.empty(R, dtype=torch.float32, device=logits.device)
    partials = torch.empty(GRID, dtype=torch.float32, device=logits.device)
    scalar_out = torch.empty(1, dtype=torch.float32, device=logits.device)

    # Launch row‑wise cross‑entropy kernel
    grid_row = (R,)
    cross_entropy_row_kernel[grid_row](
        logits, targets, loss_row,
        N, BLOCK_SIZE,
        num_warps=8,
    )

    # Launch first reduction stage (sum chunks of rows)
    # We use BLOCK_SIZE_RED1 = ROWS_PER_PROG (256) to load all at once
    BLOCK_SIZE_RED1 = ROWS_PER_PROG   # 256
    grid_red1 = (GRID,)
    reduce_stage1_kernel[grid_red1](
        loss_row, partials,
        ROWS_PER_PROG, BLOCK_SIZE_RED1,
        num_warps=4,
    )

    # Launch second reduction stage (final sum and mean)
    BLOCK_SIZE_RED2 = 32  # ≥ GRID
    grid_red2 = (1,)
    reduce_stage2_kernel[grid_red2](
        partials, scalar_out,
        R, GRID, BLOCK_SIZE_RED2,
        num_warps=4,
    )

    return scalar_out.squeeze(0)