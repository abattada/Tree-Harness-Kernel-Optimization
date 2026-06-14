import torch
import triton
import triton.language as tl


@triton.jit
def kl_div_fused_kernel(
    logp_ptr,
    q_ptr,
    output_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    NUM_PROGS: tl.constexpr,
):
    """Fused kernel: each program loops over a subset of rows and columns,
    accumulating sum(q*(log q - log_p)) locally, then atomically adds to output."""
    pid = tl.program_id(0)
    total = 0.0

    # Each program handles rows with stride NUM_PROGS
    for row_idx in range(pid, N, NUM_PROGS):
        row_offset = row_idx * C
        for col_start in range(0, C, BLOCK_C):
            cols = col_start + tl.arange(0, BLOCK_C)
            mask = cols < C
            logp = tl.load(logp_ptr + row_offset + cols, mask=mask, other=0.0)
            q_vals = tl.load(q_ptr + row_offset + cols, mask=mask, other=0.0)

            # Safe log: where q > 0 use log(q), else 0
            safe_logq = tl.where(q_vals > 0.0, tl.log(q_vals), 0.0)
            term = q_vals * (safe_logq - logp)
            partial = tl.sum(term)
            total += partial

    # Atomic contribution to global output
    tl.atomic_add(output_ptr, total)


def triton_run(log_p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """KL divergence batchmean reduction: sum(q*(log q - log_p)) / batch."""
    N, C = log_p.shape
    assert q.shape == (N, C), "shape mismatch"

    # Allocate zero-initialized scalar for atomic accumulation
    output = torch.zeros((), dtype=torch.float32, device=log_p.device)

    # Choose block sizes and parallelism
    BLOCK_C = 512   # elements per column tile, tuned for vectorized loads
    NUM_PROGS = 256 # programs cover the row dimension with stride

    # Launch the fused kernel (single pass)
    kl_div_fused_kernel[(NUM_PROGS,)](
        log_p, q, output,
        N=N, C=C,
        BLOCK_C=BLOCK_C,
        NUM_PROGS=NUM_PROGS,
        num_warps=16,   # each warp handles 32 elements of BLOCK_C
    )

    # Final division for batchmean (host-side, trivial arithmetic)
    return output / N