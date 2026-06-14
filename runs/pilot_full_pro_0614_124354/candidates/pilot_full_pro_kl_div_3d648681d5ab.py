import torch
import triton
import triton.language as tl


@triton.jit
def _kl_div_stage1_kernel(
    log_p_ptr, q_ptr, partials_ptr, total_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    lp = tl.load(log_p_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    qv = tl.load(q_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # term = q * (log(q) - log_p)
    log_q = tl.log(qv)
    term = qv * (log_q - lp)

    partial_sum = tl.sum(term)
    tl.store(partials_ptr + pid, partial_sum)


@triton.jit
def _kl_div_stage2_kernel(
    partials_ptr, out_ptr, N, num_partials, BLOCK_SIZE2: tl.constexpr
):
    acc = tl.zeros([1], dtype=tl.float32)
    for off in range(0, num_partials, BLOCK_SIZE2):
        offsets = off + tl.arange(0, BLOCK_SIZE2)
        mask = offsets < num_partials
        p = tl.load(partials_ptr + offsets, mask=mask, other=0.0)
        acc += tl.sum(p)

    loss = acc / N
    tl.store(out_ptr, loss)


def triton_run(log_p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Compute KL-divergence loss with reduction='batchmean'.
    log_p: [N, D] log-probabilities (float32)
    q:     [N, D] target probabilities (float32)
    returns scalar tensor float32
    """
    log_p = log_p.contiguous()
    q = q.contiguous()
    N, D = log_p.shape
    total_elements = N * D

    out = torch.empty((), dtype=torch.float32, device=log_p.device)

    # Stage 1: partial sums per block
    BLOCK_SIZE = 4096
    grid_size = triton.cdiv(total_elements, BLOCK_SIZE)
    partials = torch.empty((grid_size,), dtype=torch.float32, device=log_p.device)
    _kl_div_stage1_kernel[(grid_size,)](
        log_p, q, partials, total_elements,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=4
    )

    # Stage 2: final reduction over partials
    BLOCK_SIZE2 = 256
    _kl_div_stage2_kernel[(1,)](
        partials, out, N, grid_size,
        BLOCK_SIZE2=BLOCK_SIZE2, num_warps=2
    )

    return out