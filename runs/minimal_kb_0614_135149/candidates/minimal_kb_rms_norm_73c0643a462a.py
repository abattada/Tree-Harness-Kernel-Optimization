import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    x_ptr, out_ptr,
    N: tl.constexpr,
    eps: tl.constexpr,
    M: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROG
    offsets = tl.arange(0, BLOCK)

    # Process multiple rows per program to amortise launch overhead
    for r in range(ROWS_PER_PROG):
        row = row_start + r
        if row < M:
            # Coalesced load of the entire row with streaming hint
            x = tl.load(x_ptr + row * N + offsets, mask=None, eviction_policy='evict_first')
            x_sq = x * x
            sum_sq = tl.sum(x_sq, axis=0)      # parallel reduction across the block
            mean_sq = sum_sq / N
            rstd = tl.rsqrt(mean_sq + eps)
            out = x * rstd
            # Store with 'evict_last' as output may be reused soon
            tl.store(out_ptr + row * N + offsets, out, mask=None, eviction_policy='evict_last')


def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous()
    M, N = x.shape   # (8192, 4096)
    out = torch.empty_like(x)

    BLOCK = N         # one full row per tile
    ROWS_PER_PROG = 32
    grid = ((M + ROWS_PER_PROG - 1) // ROWS_PER_PROG,)

    rms_norm_kernel[grid](
        x, out,
        N=N,
        eps=1e-5,
        M=M,
        BLOCK=BLOCK,
        ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=8,
        num_stages=2,
    )
    return out