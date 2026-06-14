import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Per‑row kernel: online softmax to compute logsumexp, then subtract target logit.
# One program per row; loops over chunks of N.
# ---------------------------------------------------------------------------
@triton.jit
def cross_entropy_row_kernel(
    logits_ptr,   # f32 [R, N]
    targets_ptr,  # i64 [R]
    loss_out_ptr, # f32 [R]  (per‑row loss)
    N: tl.constexpr,          # number of classes
    BLOCK_SIZE: tl.constexpr, # tile size along N
):
    pid = tl.program_id(0)
    # Load target index for this row
    target = tl.load(targets_ptr + pid)

    # online softmax: running max and sum of exp(x - max)
    m_old = tl.full([], float('-inf'), dtype=tl.float32)
    d_old = tl.full([], 0.0, dtype=tl.float32)

    row_offset = pid * N
    for start in range(0, N, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N  # keep safe, though N is a multiple of BLOCK_SIZE
        x = tl.load(logits_ptr + row_offset + offs, mask=mask, other=float('-inf'))

        m_loc = tl.max(x, axis=0)
        m_new = tl.maximum(m_old, m_loc)

        # Compute centered exp and sum; -inf -> exp(-inf)=0
        exp_centered = tl.exp(x - m_new)
        sum_exp = tl.sum(exp_centered, axis=0)

        # Correct previous sum if max changed
        d_new = d_old * tl.exp(m_old - m_new) + sum_exp

        m_old = m_new
        d_old = d_new

    logsumexp = m_old + tl.log(d_old)

    # Load the logit at the target position
    target_logit = tl.load(logits_ptr + row_offset + target)

    # Per‑row loss = logsumexp - logit[target]
    loss = logsumexp - target_logit

    tl.store(loss_out_ptr + pid, loss)


# ---------------------------------------------------------------------------
# Reduction kernel: sum all per‑row losses and divide by number of rows.
# Single program, single block reduction.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_mean_kernel(
    inp_ptr,       # f32 [R]
    out_ptr,       # f32 [1]
    R: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    total = tl.zeros([], dtype=tl.float32)
    for start in range(0, R, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < R
        vals = tl.load(inp_ptr + offs, mask=mask, other=0.0)
        total += tl.sum(vals)
    mean = total / R
    # All threads have the same value; store by thread 0
    tl.store(out_ptr, mean)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def triton_run(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    R, N = logits.shape
    assert R == targets.shape[0]
    assert logits.device == targets.device

    # Allocate per‑row losses and final scalar output
    loss_per_row = torch.empty(R, dtype=torch.float32, device=logits.device)
    scalar_out = torch.empty(1, dtype=torch.float32, device=logits.device)

    BLOCK_SIZE_N = 1024
    grid_row = (R,)
    cross_entropy_row_kernel[grid_row](
        logits, targets, loss_per_row,
        N, BLOCK_SIZE_N,
        num_warps=8,
        num_stages=4,
    )

    BLOCK_SIZE_R = 1024
    grid_red = (1,)
    reduce_mean_kernel[grid_red](
        loss_per_row, scalar_out,
        R, BLOCK_SIZE_R,
        num_warps=4,
        num_stages=2,
    )

    return scalar_out.squeeze(0)