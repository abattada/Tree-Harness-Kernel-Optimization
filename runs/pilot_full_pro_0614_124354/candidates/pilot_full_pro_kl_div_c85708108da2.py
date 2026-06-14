import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice  # allowed, not used in this kernel

# ---------------------------------------------------------------------------
# Tunable constants (clean knobs for later tuning)
# ---------------------------------------------------------------------------
BLOCK_COL = 4096          # number of columns processed per program iteration
NUM_WARPS_ROW = 8         # warps for the first reduction (row-wise partial sums)
BLOCK_REDUCE = 8192       # reduction block size for the final sum (cover all rows)
NUM_WARPS_REDUCE = 8      # warps for the final reduction kernel

# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
@triton.jit
def _kl_div_row_kernel(
    log_p_ptr,                  # f32[B,N]
    q_ptr,
    partials_ptr,               # f32[B]
    B,
    N,
    BLOCK_COL: tl.constexpr,
):
    """
    Each program sums one row:   sum_j  q[row, j] * (log(q[row,j]) - log_p[row,j])
    Writes the row sum into partials_ptr[row].
    """
    row_idx = tl.program_id(0)
    if row_idx >= B:
        return

    # Base pointers for this row
    off_logp = log_p_ptr + row_idx * N
    off_q = q_ptr + row_idx * N

    # Accumulator in fp32
    acc = tl.zeros([1], dtype=tl.float32)

    # Column offsets for vectorised access
    col_offs = tl.arange(0, BLOCK_COL)

    for col_start in range(0, N, BLOCK_COL):
        cols = col_start + col_offs
        mask = cols < N

        # Load a contiguous block
        logp = tl.load(off_logp + cols, mask=mask, other=0.0)
        q_val = tl.load(off_q + cols, mask=mask, other=0.0)

        # Safe log(q):  0 * log(0)  must be  0
        log_q = tl.where(q_val > 0.0, tl.log(q_val), 0.0)
        diff = log_q - logp
        contrib = q_val * diff

        # Only valid elements contribute
        contrib = tl.where(mask, contrib, 0.0)

        acc += tl.sum(contrib)

    tl.store(partials_ptr + row_idx, acc)


@triton.jit
def _reduce_sum_kernel(
    partials_ptr,               # f32[B]
    out_ptr,                    # f32[1] – will hold the total sum
    B,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Single‑block reduction: sums the partial row sums into out_ptr.
    """
    offs = tl.arange(0, BLOCK_SIZE)
    total = tl.zeros([1], dtype=tl.float32)

    for start in range(0, B, BLOCK_SIZE):
        idx = start + offs
        mask = idx < B
        vals = tl.load(partials_ptr + idx, mask=mask, other=0.0)
        total += tl.sum(tl.where(mask, vals, 0.0))

    tl.store(out_ptr, total)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def triton_run(log_p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence:   sum(q * (log q - log_p)) / batch    (batchmean)

    Args:
        log_p: f32[8192, 8192]   log-probabilities
        q:     f32[8192, 8192]   probabilities

    Returns:
        f32[]   scalar tensor
    """
    B, N = log_p.shape
    assert q.shape == (B, N), "input shapes must match"
    device = log_p.device

    # 1st stage: partial row sums → buffer of size B
    partials = torch.empty(B, dtype=torch.float32, device=device)
    grid_row = (B,)
    _kl_div_row_kernel[grid_row](
        log_p, q, partials, B, N,
        BLOCK_COL=BLOCK_COL,
        num_warps=NUM_WARPS_ROW,
    )

    # 2nd stage: sum partials into a scalar
    out = torch.zeros(1, dtype=torch.float32, device=device)
    _reduce_sum_kernel[(1,)](
        partials, out, B,
        BLOCK_SIZE=BLOCK_REDUCE,
        num_warps=NUM_WARPS_REDUCE,
    )

    # batchmean = total_sum / B
    result = out / B
    return result.squeeze()   # 0‑dim tensor