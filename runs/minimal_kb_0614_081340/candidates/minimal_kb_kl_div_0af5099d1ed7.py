import torch
import triton
import triton.language as tl

BLOCK_COLS = 1024

@triton.jit
def kl_div_row_kernel(
    log_p_ptr,
    q_ptr,
    out_ptr,
    n_cols: tl.constexpr,
    batch_size: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid
    # Linear offset for the start of this row
    row_offset = row * n_cols
    sum_loss = 0.0

    # Process columns in blocks
    for start in range(0, n_cols, BLOCK_COLS):
        offs = start + tl.arange(0, BLOCK_COLS)
        mask = offs < n_cols

        lp = tl.load(log_p_ptr + row_offset + offs, mask=mask, other=0.0)
        qv = tl.load(q_ptr + row_offset + offs, mask=mask, other=0.0)

        # KL divergence contribution: q * (log(q) - log_p), zero when q == 0
        loss = tl.where(qv > 0, qv * (tl.log(qv) - lp), 0.0)
        sum_loss += tl.sum(loss)

    # Atomic accumulate the row sum into the global output
    tl.atomic_add(out_ptr, sum_loss)

def triton_run(log_p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence with reduction='batchmean'.

    log_p: (B, C) log-probabilities
    q: (B, C) target probabilities
    returns: scalar
    """
    assert log_p.is_contiguous() and q.is_contiguous()
    B, C = log_p.shape
    assert q.shape == (B, C)

    out = torch.zeros(1, dtype=torch.float32, device=log_p.device)

    grid = (B,)
    kl_div_row_kernel[grid](
        log_p,
        q,
        out,
        n_cols=C,
        batch_size=B,
    )

    result = out / B
    return result