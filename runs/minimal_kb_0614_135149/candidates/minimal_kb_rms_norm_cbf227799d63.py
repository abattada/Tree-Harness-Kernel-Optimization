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
    offsets = tl.arange(0, BLOCK)

    # Load the full row with streaming hint
    x = tl.load(x_ptr + row * N + offsets, mask=None, eviction_policy='evict_first')

    # Compute RMS via single-pass sum of squares
    x_sq = x * x
    sum_sq = tl.sum(x_sq, axis=0)
    mean_sq = sum_sq / N
    rstd = tl.rsqrt(mean_sq + eps)

    out = x * rstd
    # Store the normalised row; keep in cache as it may be reused
    tl.store(out_ptr + row * N + offsets, out, mask=None, eviction_policy='evict_last')


def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous()
    M, N = x.shape
    out = torch.empty_like(x)

    # Each block handles exactly one row
    grid = (M,)
    BLOCK = N

    rms_norm_kernel[grid](
        x, out,
        N=N,
        eps=1e-5,
        BLOCK=BLOCK,
        num_warps=16,   # increased from 8 to better hide memory latency
        num_stages=2,   # enable pipelining of the next row's load
    )
    return out