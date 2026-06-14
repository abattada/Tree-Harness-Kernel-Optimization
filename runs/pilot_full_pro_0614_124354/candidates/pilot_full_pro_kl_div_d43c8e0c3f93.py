import torch
import triton
import triton.language as tl

@triton.jit
def _kl_div_sum_kernel(log_p_ptr, q_ptr, partial_sum_ptr,
                       BLOCK_SIZE: tl.constexpr,
                       VEC_SIZE: tl.constexpr):
    """Compute per-tile partial sum of KL divergence terms."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    acc = 0.0
    for i in range(0, BLOCK_SIZE, VEC_SIZE):
        idx = block_start + i + tl.arange(0, VEC_SIZE)
        q_vals = tl.load(q_ptr + idx)
        log_p_vals = tl.load(log_p_ptr + idx)

        # safe log: 0 * log(0) → 0
        safe_log_q = tl.where(q_vals > 0.0, tl.log(q_vals), 0.0)
        term = tl.where(q_vals > 0.0,
                       q_vals * (safe_log_q - log_p_vals),
                       0.0)

        acc += tl.sum(term)  # sum across threads inside the program

    tl.store(partial_sum_ptr + pid, acc)


@triton.jit
def _kl_div_reduce_kernel(partial_sum_ptr, out_ptr,
                          num_partials: tl.constexpr,
                          BATCH_SIZE: tl.constexpr,
                          BLOCK_REDUCE: tl.constexpr):
    """Final reduction: sum all partials and divide by batch size."""
    pid = tl.program_id(0)
    acc = 0.0
    for i in range(0, num_partials, BLOCK_REDUCE):
        idx = i + tl.arange(0, BLOCK_REDUCE)
        mask = idx < num_partials
        vals = tl.load(partial_sum_ptr + idx, mask=mask, other=0.0)
        acc += tl.sum(vals)

    result = acc / BATCH_SIZE
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
    B = log_p.shape[0]                 # 8192
    total_elements = log_p.numel()    # 8192 * 8192

    # Tile configuration
    BLOCK_SIZE = 4096
    VEC_SIZE = 4
    assert total_elements % BLOCK_SIZE == 0, (
        "total elements must be divisible by BLOCK_SIZE"
    )
    num_partials = total_elements // BLOCK_SIZE

    # Stage 1: per-tile partial sums
    partials = torch.empty(num_partials, dtype=torch.float32, device=log_p.device)
    grid1 = (num_partials,)
    _kl_div_sum_kernel[grid1](
        log_p, q, partials,
        BLOCK_SIZE=BLOCK_SIZE, VEC_SIZE=VEC_SIZE,
        num_warps=8,
    )

    # Stage 2: final reduction
    out = torch.empty(1, dtype=torch.float32, device=log_p.device)
    BLOCK_REDUCE = 1024
    grid2 = (1,)
    _kl_div_reduce_kernel[grid2](
        partials, out,
        num_partials=num_partials,
        BATCH_SIZE=B,
        BLOCK_REDUCE=BLOCK_REDUCE,
        num_warps=1,
    )

    return out.squeeze()