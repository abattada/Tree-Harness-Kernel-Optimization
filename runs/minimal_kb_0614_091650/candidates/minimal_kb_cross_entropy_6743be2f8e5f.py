import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 1: per-row online softmax + NLL loss
# Each program processes one row in chunks.
# ---------------------------------------------------------------------------
@triton.jit
def cross_entropy_row_kernel(
    logits_ptr,        # [rows, cols] f32
    targets_ptr,       # [rows] i64
    loss_row_ptr,      # [rows] f32 (output)
    rows: tl.constexpr,
    cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)          # row index
    target = tl.load(targets_ptr + pid)

    m_old = tl.full([], float('-inf'), dtype=tl.float32)
    d_old = tl.full([], 0.0, dtype=tl.float32)

    # Loop over column blocks
    for start in range(0, cols, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < cols
        x = tl.load(logits_ptr + pid * cols + offs, mask=mask, other=float('-inf'))

        # Online softmax update
        m_loc = tl.max(x, axis=0)
        m_new = tl.maximum(m_old, m_loc)
        exp_centered = tl.exp(x - m_new)
        sum_exp = tl.sum(exp_centered, axis=0)
        d_new = d_old * tl.exp(m_old - m_new) + sum_exp
        m_old = m_new
        d_old = d_new

    logsumexp = m_old + tl.log(d_old)
    target_logit = tl.load(logits_ptr + pid * cols + target)
    loss = logsumexp - target_logit
    tl.store(loss_row_ptr + pid, loss)


# ---------------------------------------------------------------------------
# Kernel 2: reduction – sum all per‑row losses and compute mean
# We use a single program to sum over rows in a grid‑stride loop.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_mean_kernel(
    inp_ptr,       # [rows] f32
    out_ptr,       # [1] f32 (output)
    rows: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    total = tl.zeros([], dtype=tl.float32)
    for start in range(0, rows, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < rows
        vals = tl.load(inp_ptr + offs, mask=mask, other=0.0)
        total += tl.sum(vals)
    # All threads have the same total; write via thread 0
    tl.store(out_ptr, total / rows.to(tl.float32))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def triton_run(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    rows, cols = logits.shape
    assert rows == targets.shape[0]
    device = logits.device

    # Allocate per-row loss buffer
    loss_per_row = torch.empty(rows, dtype=torch.float32, device=device)
    # Allocate scalar output
    scalar_out = torch.empty(1, dtype=torch.float32, device=device)

    # Launch parameters (tunable)
    BLOCK_SIZE = 1024   # column block size – tune for performance
    NUM_WARPS_ROW = 8   # warps per program for row kernel
    NUM_WARPS_RED = 4   # warps for reduction kernel

    grid_row = (rows,)   # one program per row
    cross_entropy_row_kernel[grid_row](
        logits, targets, loss_per_row,
        rows, cols, BLOCK_SIZE,
        num_warps=NUM_WARPS_ROW,
    )

    # Reduction: single program sums all rows
    grid_red = (1,)
    reduce_mean_kernel[grid_red](
        loss_per_row, scalar_out,
        rows, BLOCK_SIZE,
        num_warps=NUM_WARPS_RED,
    )

    return scalar_out.squeeze(0)