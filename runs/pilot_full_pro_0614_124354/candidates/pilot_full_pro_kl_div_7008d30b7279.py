import torch
import triton
import triton.language as tl


@triton.jit
def kl_div_row_kernel(
    logp_ptr, q_ptr, row_sum_ptr, N, C, BLOCK_C: tl.constexpr
):
    """
    Per-row KL divergence sum: sum_over_cols( q * (log q - logp) ).
    Tuned BLOCK_C trades off loop overhead vs register pressure.
    """
    pid = tl.program_id(0)
    row = pid
    if row < N:
        offset = row * C
        acc = 0.0
        for start in range(0, C, BLOCK_C):
            cols = start + tl.arange(0, BLOCK_C)
            mask = cols < C
            logp = tl.load(logp_ptr + offset + cols, mask=mask, other=0.0)
            q = tl.load(q_ptr + offset + cols, mask=mask, other=0.0)
            safe_logq = tl.where(q > 0.0, tl.log(q), 0.0)
            term = q * (safe_logq - logp)
            acc += tl.sum(term)
        tl.store(row_sum_ptr + row, acc)


@triton.jit
def reduce_sum_kernel(row_sum_ptr, output_ptr, N, BLOCK_N: tl.constexpr):
    """Sum over rows and divide by N (batchmean)."""
    total = 0.0
    for start in range(0, N, BLOCK_N):
        idx = start + tl.arange(0, BLOCK_N)
        mask = idx < N
        vals = tl.load(row_sum_ptr + idx, mask=mask, other=0.0)
        total += tl.sum(vals)
    result = total / N
    tl.store(output_ptr, result)


def triton_run(log_p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """KL divergence: batchmean reduction  sum(q*(log q - log_p)) / N."""
    N, C = log_p.shape
    assert q.shape == (N, C), "shape mismatch"

    row_sum = torch.empty(N, dtype=torch.float32, device=log_p.device)

    # Tuned block size: 1024 balances occupancy (512 threads, 2 elts/thread)
    # and halves the loop count vs 512, keeping register pressure low.
    BLOCK_C = 1024
    kl_div_row_kernel[(N,)](
        log_p, q, row_sum, N, C,
        BLOCK_C=BLOCK_C,
        num_warps=16,
    )

    output = torch.empty((), dtype=torch.float32, device=log_p.device)
    BLOCK_N = 512
    reduce_sum_kernel[(1,)](
        row_sum, output, N,
        BLOCK_N=BLOCK_N,
        num_warps=16,
    )

    return output