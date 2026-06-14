import torch
import triton
import triton.language as tl

@triton.jit
def _kl_div_row_sum_kernel(log_p_ptr, q_ptr, row_sums_ptr, N: tl.constexpr):
    """Compute KL divergence sum for a whole row."""
    pid = tl.program_id(0)
    offs = pid * N + tl.arange(0, N)
    q_vals = tl.load(q_ptr + offs)
    log_p_vals = tl.load(log_p_ptr + offs)

    # Safe KL: q * (log(q) - log_p), with 0 * log(0) = 0
    safe_log_q = tl.where(q_vals > 0.0, tl.log(q_vals), 0.0)
    term = tl.where(q_vals > 0.0, q_vals * (safe_log_q - log_p_vals), 0.0)

    row_sum = tl.sum(term)  # sum across all threads in the program
    tl.store(row_sums_ptr + pid, row_sum)


@triton.jit
def _kl_div_reduce_kernel(row_sums_ptr, out_ptr, B: tl.constexpr, BLOCK: tl.constexpr):
    """Final reduction: sum all row sums and divide by batch size."""
    pid = tl.program_id(0)
    acc = 0.0
    offs = tl.arange(0, BLOCK)
    for start in range(0, B, BLOCK):
        vals = tl.load(row_sums_ptr + start + offs, mask=start + offs < B, other=0.0)
        acc += tl.sum(vals)

    result = acc / B
    if pid == 0:
        tl.store(out_ptr, result)


def triton_run(log_p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence with batchmean reduction.

    Args:
        log_p: log-probabilities, shape [8192, 8192], float32
        q:     target probabilities, same shape, float32

    Returns:
        scalar tensor with KL divergence (batchmean)
    """
    B, N = log_p.shape
    assert B == 8192 and N == 8192, "Expected [8192, 8192] inputs"

    # Stage 1: per‑row partial sums
    row_sums = torch.empty(B, dtype=torch.float32, device=log_p.device)
    grid1 = (B,)
    _kl_div_row_sum_kernel[grid1](
        log_p, q, row_sums,
        N=N,
        num_warps=16,
    )

    # Stage 2: final reduction
    out = torch.empty(1, dtype=torch.float32, device=log_p.device)
    BLOCK_REDUCE = 1024
    grid2 = (1,)
    _kl_div_reduce_kernel[grid2](
        row_sums, out,
        B=B,
        BLOCK=BLOCK_REDUCE,
        num_warps=1,
    )

    return out.squeeze()