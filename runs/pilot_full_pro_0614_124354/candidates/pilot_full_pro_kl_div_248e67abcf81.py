import torch
import triton
import triton.language as tl


@triton.jit
def kl_div_row_kernel(
    logp_ptr,
    q_ptr,
    row_sum_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """
    Per-row reduction: sum over C of q*(log q - logp).
    Specialized for exact multiple of BLOCK_C – no boundary mask.
    Each thread processes several elements per loop iteration.
    """
    tl.static_assert(C % BLOCK_C == 0)
    pid = tl.program_id(0)
    row = pid
    if row < N:
        offset = row * C
        acc = 0.0
        # C is exactly divisible by BLOCK_C → mask is never needed
        for start in range(0, C, BLOCK_C):
            cols = start + tl.arange(0, BLOCK_C)
            # Streaming loads: data is read once per row → evict_first
            logp = tl.load(
                logp_ptr + offset + cols,
                eviction_policy="evict_first",
            )
            q = tl.load(
                q_ptr + offset + cols,
                eviction_policy="evict_first",
            )
            safe_logq = tl.where(q > 0.0, tl.log(q), 0.0)
            term = q * (safe_logq - logp)
            acc += tl.sum(term)
        tl.store(row_sum_ptr + row, acc)


@triton.jit
def reduce_sum_kernel(
    row_sum_ptr,
    output_ptr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Final scalar reduction: sum per-row values and divide by N."""
    tl.static_assert(N % BLOCK_N == 0)
    total = 0.0
    for start in range(0, N, BLOCK_N):
        idx = start + tl.arange(0, BLOCK_N)
        # row_sum fits in L2, let it be cached (default policy)
        vals = tl.load(row_sum_ptr + idx)
        total += tl.sum(vals)
    result = total / N
    tl.store(output_ptr, result)


def triton_run(log_p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    KL divergence batchmean: sum(q*(log q - log_p)) / batch.
    Vertorised, mask-free kernels tuned for 8192 × 8192 input.
    """
    N, C = log_p.shape
    assert q.shape == (N, C), "shape mismatch"

    # intermediate per-row partial sums
    row_sum = torch.empty(N, dtype=torch.float32, device=log_p.device)
    output = torch.empty((), dtype=torch.float32, device=log_p.device)

    # Process 2048 elements per loop iteration → each thread handles 4 elements
    BLOCK_C = 2048
    # 8192 rows, each row processed by one program
    kl_div_row_kernel[(N,)](
        log_p, q, row_sum,
        N=N, C=C, BLOCK_C=BLOCK_C,
        num_warps=16,          # 512 threads
    )

    # Final reduction: 2048 rows per loop iteration
    BLOCK_N = 2048
    reduce_sum_kernel[(1,)](
        row_sum, output,
        N=N, BLOCK_N=BLOCK_N,
        num_warps=16,
    )

    return output