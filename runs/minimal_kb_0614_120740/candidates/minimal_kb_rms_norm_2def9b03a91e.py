import torch
import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(
    x_ptr, out_ptr,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    """
    Each program processes ROWS_PER_PROG consecutive rows.
    """
    row_start = tl.program_id(0) * ROWS_PER_PROG
    offsets = tl.arange(0, BLOCK)

    for r in range(ROWS_PER_PROG):
        row = row_start + r
        x_row_ptr = x_ptr + row * N
        out_row_ptr = out_ptr + row * N

        # Load entire row (N == BLOCK, so no mask needed)
        x = tl.load(x_row_ptr + offsets, eviction_policy='evict_first')

        # Single-pass RMS
        x_sq = x * x
        sum_sq = tl.sum(x_sq, axis=0)
        mean_sq = sum_sq / N
        rstd = tl.rsqrt(mean_sq + eps)

        out = x * rstd
        tl.store(out_row_ptr + offsets, out, eviction_policy='evict_last')


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    x: (8192, 4096) float32 contiguous.
    Returns: same shape, rms-normalized.
    """
    assert x.is_contiguous()
    M, N = x.shape
    out = torch.empty_like(x)

    BLOCK = N  # Exact fit, no masking
    ROWS_PER_PROG = 4  # Reduce launch overhead; 8192/4 = 2048 programs
    assert M % ROWS_PER_PROG == 0, "M must be divisible by ROWS_PER_PROG"

    grid = (M // ROWS_PER_PROG,)
    rms_norm_kernel[grid](
        x, out,
        N=N,
        eps=1e-5,
        BLOCK=BLOCK,
        ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=8,
        num_stages=1,
    )
    return out