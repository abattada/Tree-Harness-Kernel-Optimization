import torch
import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(
    x_ptr, out_ptr,
    N: tl.constexpr,
    eps: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    One program processes multiple consecutive rows to amortize grid launch
    overhead and increase work per program, improving occupancy.
    """
    pid = tl.program_id(0)
    base_row = pid * ROWS_PER_PROG
    offsets = tl.arange(0, BLOCK)  # full row length, no mask needed

    for r in range(ROWS_PER_PROG):
        row = base_row + r
        x_row_ptr = x_ptr + row * N
        out_row_ptr = out_ptr + row * N

        # Load row (streaming → evict first)
        x = tl.load(x_row_ptr + offsets, eviction_policy='evict_first')

        # One-pass RMS: sum of squares, mean, rsqrt
        sum_sq = tl.sum(x * x, axis=0)
        mean_sq = sum_sq / N
        rstd = tl.rsqrt(mean_sq + eps)

        out = x * rstd
        # Store output (may be reused → evict last)
        tl.store(out_row_ptr + offsets, out, eviction_policy='evict_last')


def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous()
    M, N = x.shape
    out = torch.empty_like(x)

    # Process ROWS_PER_PROG rows per program (divisible by M)
    ROWS_PER_PROG = 64   # → grid = 8192/64 = 128 programs
    BLOCK = N
    grid = (M // ROWS_PER_PROG,)

    rms_norm_kernel[grid](
        x, out,
        N=N,
        eps=1e-5,
        ROWS_PER_PROG=ROWS_PER_PROG,
        BLOCK=BLOCK,
        num_warps=8,
        num_stages=1,
    )
    return out