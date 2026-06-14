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
    # One program per row
    row = tl.program_id(0)
    x_row_ptr = x_ptr + row * N
    out_row_ptr = out_ptr + row * N

    # Contiguous columns, no mask needed since N == BLOCK
    offsets = tl.arange(0, BLOCK)
    x = tl.load(x_row_ptr + offsets, eviction_policy='evict_first')

    # Single-pass RMS: sum of squares, mean, rsqrt
    x_sq = x * x
    sum_sq = tl.sum(x_sq, axis=0)
    mean_sq = sum_sq / N
    rstd = tl.rsqrt(mean_sq + eps)

    out = x * rstd
    tl.store(out_row_ptr + offsets, out, eviction_policy='evict_last')


def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous()
    M, N = x.shape
    out = torch.empty_like(x)

    BLOCK = N  # One tile per row
    grid = (M,)

    rms_norm_kernel[grid](
        x, out,
        N=N,
        eps=1e-5,
        BLOCK=BLOCK,
        num_warps=4,          # Reduced from 8 to try more blocks per SM
        num_stages=1,
    )
    return out