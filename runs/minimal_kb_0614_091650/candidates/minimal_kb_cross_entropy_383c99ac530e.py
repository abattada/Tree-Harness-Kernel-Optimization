import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 1: per‑row online softmax + cross-entropy loss.
# Each program processes one row of shape [N] (N is multiple of BLOCK_SIZE).
# ---------------------------------------------------------------------------
@triton.jit
def cross_entropy_row_kernel(
    logits_ptr,   # f32 [R, N]
    targets_ptr,  # i64 [R]
    loss_row_ptr, # f32 [R]   (output per‑row loss)
    N: tl.constexpr,             # number of classes (multiple of BLOCK_SIZE)
    BLOCK_SIZE: tl.constexpr     # tile along the N dimension
):
    pid = tl.program_id(0)                     # row index
    target = tl.load(targets_ptr + pid)        # scalar i64

    # Online softmax: compute max and sum of exp in a single pass
    m_old = tl.full([], float('-inf'), dtype=tl.float32)
    d_old = tl.full([], 0.0, dtype=tl.float32)

    # N is a multiple of BLOCK_SIZE, so all blocks are full.
    for start in range(0, N, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        # Load without mask (since BLOCK_SIZE divides N)
        x = tl.load(logits_ptr + pid * N + offs,
                    eviction_policy='evict_last')

        # Local max and sum of exponentials after centering with m_new
        m_loc = tl.max(x, axis=0)
        m_new = tl.maximum(m_old, m_loc)
        exp_centered = tl.exp(x - m_new)
        sum_exp = tl.sum(exp_centered, axis=0)

        # Online update: scale previous contributions and add new
        d_new = d_old * tl.exp(m_old - m_new) + sum_exp
        m_old = m_new
        d_old = d_new

    # logsumexp = m_old + log(d_old)
    logsumexp = m_old + tl.log(d_old)

    # Load the logit at the target position
    target_logit = tl.load(logits_ptr + pid * N + target)

    # Per‑row loss = logsumexp - logit[target]
    loss = logsumexp - target_logit
    tl.store(loss_row_ptr + pid, loss, eviction_policy='evict_first')


# ---------------------------------------------------------------------------
# Kernel 2: single‑stage reduction – sum all per‑row losses and divide by R.
# Only one program is launched; it loops over all rows.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_all_kernel(
    loss_row_ptr,  # f32 [R]
    scalar_ptr,    # f32 [1]
    R: tl.constexpr,          # number of rows
    BLOCK_SIZE: tl.constexpr  # tile for the reduction loop
):
    total = tl.full([], 0.0, dtype=tl.float32)
    for start in range(0, R, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        # Last block may be incomplete (R may not be a multiple)
        mask = offs < R
        vals = tl.load(loss_row_ptr + offs, mask=mask, other=0.0,
                       eviction_policy='evict_first')
        total += tl.sum(vals)
    mean = total / R
    # All threads have the same value; store by thread 0
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
    assert N % 1024 == 0, "N must be a multiple of 1024 for this implementation"

    # Allocate intermediate and output buffers
    loss_row = torch.empty(R, dtype=torch.float32, device=logits.device)
    scalar_out = torch.empty(1, dtype=torch.float32, device=logits.device)

    # Launch row‑wise cross‑entropy kernel
    BLOCK_SIZE_ROW = 1024
    grid_row = (R,)
    cross_entropy_row_kernel[grid_row](
        logits, targets, loss_row,
        N, BLOCK_SIZE_ROW,
        num_warps=8,
    )

    # Launch single‑stage reduction kernel
    BLOCK_SIZE_RED = 1024
    grid_red = (1,)
    reduce_all_kernel[grid_red](
        loss_row, scalar_out,
        R, BLOCK_SIZE_RED,
        num_warps=4,
    )

    return scalar_out.squeeze(0)