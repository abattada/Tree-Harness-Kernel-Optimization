import torch
import triton
import triton.language as tl


@triton.jit
def kl_div_fused_kernel(
    logp_ptr,
    q_ptr,
    row_sum_ptr,
    N,
    C,
    BLOCK_C: tl.constexpr,
):
    """
    Fused per‑row KL divergence:
    For one row, compute sum_{c}(q[c] * (log(q[c]) - log_p[c])).
    Each program handles exactly one row; the whole elementwise
    operation and its reduction happen inside a single register loop,
    avoiding any intermediate global writes.
    """
    row = tl.program_id(0)
    if row < N:
        offset = row * C
        acc = tl.zeros((1,), dtype=tl.float32)
        # Tile over columns, all in registers.
        for c_start in range(0, C, BLOCK_C):
            cols = c_start + tl.arange(0, BLOCK_C)
            # Mask and contiguous hints help the compiler generate
            # wide, coalesced vector loads.
            mask = cols < C
            logp = tl.load(
                logp_ptr + offset + cols,
                mask=mask,
                other=0.0,
            )
            q = tl.load(
                q_ptr + offset + cols,
                mask=mask,
                other=0.0,
            )
            # Guard log(0) while keeping the loop in fast‑math regime.
            safe_logq = tl.where(q > 0.0, tl.log(q), 0.0)
            term = q * (safe_logq - logp)
            acc += tl.sum(term)
        tl.store(row_sum_ptr + row, acc)


@triton.jit
def reduce_mean_kernel(
    row_sum_ptr,
    output_ptr,
    N,
    BLOCK_N: tl.constexpr,
):
    """
    Final scalar reduction: sum row_sum and divide by N to obtain
    the batchmean.  Also a single kernel to avoid any host‑side
    reduction (required by the operator specification).
    """
    total = tl.zeros((1,), dtype=tl.float32)
    for n_start in range(0, N, BLOCK_N):
        idx = n_start + tl.arange(0, BLOCK_N)
        mask = idx < N
        vals = tl.load(row_sum_ptr + idx, mask=mask, other=0.0)
        total += tl.sum(vals)
    result = total / N.to(tl.float32)
    tl.store(output_ptr, result)


def triton_run(log_p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    KL divergence with batchmean reduction:
       KL = (1/N) * sum_{n,c} q[n,c] * (log(q[n,c]) - log_p[n,c])
    Inputs:
      log_p : f32[N, C]   log probabilities (already log‑softmax)
      q     : f32[N, C]   target distribution
    Returns:
      scalar f32 tensor with the batchmean KL divergence.
    """
    N, C = log_p.shape
    assert q.shape == (N, C), "shape mismatch"

    # Intermediate buffer for per‑row partial sums.
    row_sum = torch.empty(N, dtype=torch.float32, device=log_p.device)

    # Choose a large tile size to saturate memory bandwidth.
    # Blackwell’s 128‑byte cache line → 32 float per line, so multiples
    # of 32 are ideal.  512 is a good default (16 warps, 1 element/thread).
    BLOCK_C = 512
    kl_div_fused_kernel[(N,)](
        log_p, q, row_sum, N, C,
        BLOCK_C=BLOCK_C,
        num_warps=16,
    )

    output = torch.empty((), dtype=torch.float32, device=log_p.device)
    BLOCK_N = 512
    reduce_mean_kernel[(1,)](
        row_sum, output, N,
        BLOCK_N=BLOCK_N,
        num_warps=16,
    )
    return output