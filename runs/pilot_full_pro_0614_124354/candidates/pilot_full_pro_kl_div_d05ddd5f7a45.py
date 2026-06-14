import torch
import triton
import triton.language as tl


@triton.jit
def kl_div_row_kernel(
    logp_ptr,
    q_ptr,
    row_sum_ptr,
    N,
    C,
    BLOCK_SIZE: tl.constexpr,
    ELEMS_PER_THREAD: tl.constexpr,
):
    """Per-row sum of q*(log q - log_p) using vectorized loads."""
    pid = tl.program_id(0)
    row = pid
    if row >= N:
        return

    base = row * C
    acc = 0.0

    CHUNK_SIZE = BLOCK_SIZE * ELEMS_PER_THREAD
    # create 2D offset pattern: (BLOCK_SIZE, ELEMS_PER_THREAD)
    off_thread = tl.arange(0, BLOCK_SIZE)[:, None] * ELEMS_PER_THREAD
    off_elem = tl.arange(0, ELEMS_PER_THREAD)[None, :]
    off = off_thread + off_elem

    for start in range(0, C, CHUNK_SIZE):
        cols = start + off
        mask = cols < C
        logp = tl.load(logp_ptr + base + cols, mask=mask, other=0.0)
        q = tl.load(q_ptr + base + cols, mask=mask, other=0.0)

        # safe log to avoid log(0) → -inf * 0 = NaN
        safe_logq = tl.where(q > 0.0, tl.log(q), 0.0)
        term = q * (safe_logq - logp)
        acc += tl.sum(term)

    tl.store(row_sum_ptr + row, acc)


@triton.jit
def reduce_sum_kernel(
    row_sum_ptr,
    output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
    ELEMS_PER_THREAD: tl.constexpr,
):
    """Vectorized sum over rows, then divide by N for batchmean."""
    total = 0.0
    CHUNK_SIZE = BLOCK_SIZE * ELEMS_PER_THREAD
    off_thread = tl.arange(0, BLOCK_SIZE)[:, None] * ELEMS_PER_THREAD
    off_elem = tl.arange(0, ELEMS_PER_THREAD)[None, :]
    off = off_thread + off_elem

    for start in range(0, N, CHUNK_SIZE):
        idx = start + off
        mask = idx < N
        vals = tl.load(row_sum_ptr + idx, mask=mask, other=0.0)
        total += tl.sum(vals)

    tl.store(output_ptr, total / N)


def triton_run(log_p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    N, C = log_p.shape
    assert q.shape == (N, C), "shape mismatch"

    row_sum = torch.empty(N, dtype=torch.float32, device=log_p.device)

    BLOCK_SIZE = 256
    ELEMS_PER_THREAD = 4
    num_warps = BLOCK_SIZE // 32  # 8

    kl_div_row_kernel[(N,)](
        log_p, q, row_sum, N, C,
        BLOCK_SIZE=BLOCK_SIZE,
        ELEMS_PER_THREAD=ELEMS_PER_THREAD,
        num_warps=num_warps,
    )

    output = torch.empty((), dtype=torch.float32, device=log_p.device)
    reduce_sum_kernel[(1,)](
        row_sum, output, N,
        BLOCK_SIZE=BLOCK_SIZE,
        ELEMS_PER_THREAD=ELEMS_PER_THREAD,
        num_warps=num_warps,
    )

    return output