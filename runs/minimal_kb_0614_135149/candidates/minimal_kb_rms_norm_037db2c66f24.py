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
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROG

    offsets = tl.arange(0, BLOCK)
    for i in tl.static_range(ROWS_PER_PROG):
        row = row_start + i
        x_row_ptr = x_ptr + row * N
        out_row_ptr = out_ptr + row * N

        # Load the row with streaming hint
        x = tl.load(x_row_ptr + offsets, eviction_policy='evict_first')
        x_sq = x * x
        sum_sq = tl.sum(x_sq, axis=0)
        mean_sq = sum_sq / N
        rstd = tl.rsqrt(mean_sq + eps)
        out = x * rstd
        # Store the normalized row with reuse hint
        tl.store(out_row_ptr + offsets, out, eviction_policy='evict_last')


def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous()
    M, N = x.shape
    out = torch.empty_like(x)

    BLOCK = N
    ROWS_PER_PROG = 2
    assert M % ROWS_PER_PROG == 0, "M must be divisible by ROWS_PER_PROG"
    grid = (M // ROWS_PER_PROG,)

    rms_norm_kernel[grid](
        x, out,
        N=N,
        eps=1e-5,
        BLOCK=BLOCK,
        ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=8,
        num_stages=2,
    )
    return out