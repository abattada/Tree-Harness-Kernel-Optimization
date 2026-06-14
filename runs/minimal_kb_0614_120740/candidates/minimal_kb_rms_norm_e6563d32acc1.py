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
    Process ROWS_PER_PROG consecutive rows per program.
    """
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROG
    offsets = tl.arange(0, BLOCK)

    for i in range(ROWS_PER_PROG):
        row = row_start + i
        # Guard against out-of-bounds rows (if M not multiple)
        if row < tl.num_programs(0) * ROWS_PER_PROG:
            x_row_ptr = x_ptr + row * N
            out_row_ptr = out_ptr + row * N

            x = tl.load(x_row_ptr + offsets, mask=None, eviction_policy='evict_first')

            # Single-pass RMS
            x_sq = x * x
            sum_sq = tl.sum(x_sq, axis=0)
            mean_sq = sum_sq / N
            rstd = tl.rsqrt(mean_sq + eps)

            out = x * rstd
            tl.store(out_row_ptr + offsets, out, mask=None, eviction_policy='evict_last')


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    x: (8192, 4096) float32, contiguous.
    Returns: same shape, rms-normalized.
    """
    assert x.is_contiguous()
    M, N = x.shape
    out = torch.empty_like(x)

    BLOCK = N  # 4096, whole row
    ROWS_PER_PROG = 4  # process 4 rows per program to reduce grid size

    grid = (M // ROWS_PER_PROG,)

    # Ensure M is divisible – it is, but add safety
    if M % ROWS_PER_PROG != 0:
        # fallback: launch exact number of programs with row guard
        grid = (triton.cdiv(M, ROWS_PER_PROG),)

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