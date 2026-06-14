import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    x_ptr, out_ptr,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    x_row_ptr = x_ptr + row * N
    out_row_ptr = out_ptr + row * N

    offsets = tl.arange(0, BLOCK)

    # Streaming load – read once, discard cache line quickly
    x = tl.load(x_row_ptr + offsets, mask=None, eviction_policy='evict_first')

    # RMS norm: x * rsqrt(mean(x^2) + eps)
    x_sq = x * x
    sum_sq = tl.sum(x_sq, axis=0)
    mean_sq = sum_sq / N
    rstd = tl.rsqrt(mean_sq + eps)

    out = x * rstd

    # Store output with evict_last – likely to be reused soon by downstream ops
    tl.store(out_row_ptr + offsets, out, mask=None, eviction_policy='evict_last')


def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous()
    M, N = x.shape  # M=8192, N=4096
    out = torch.empty_like(x)

    # One tile per row – exactly BLOCK == N columns, no boundary masking
    BLOCK = N
    grid = (M,)

    rms_norm_kernel[grid](
        x, out,
        N=N,
        eps=1e-5,
        BLOCK=BLOCK,
        num_warps=4,          # fewer warps → less reduction-tree overhead, often higher occupancy
        num_stages=1,
    )
    return out