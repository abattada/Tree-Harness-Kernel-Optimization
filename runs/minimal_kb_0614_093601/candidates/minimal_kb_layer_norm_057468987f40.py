import torch
import triton
import triton.language as tl

@triton.jit
def layer_norm_kernel(
    x_ptr, out_ptr,
    stride_x, stride_out,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Guarantee no masking needed (N == BLOCK for this shape)
    tl.static_assert(N == BLOCK)

    row = tl.program_id(0)
    x_row = x_ptr + row * stride_x
    out_row = out_ptr + row * stride_out

    cols = tl.arange(0, BLOCK)
    # Load with eviction hint: input is streaming, evict first
    x = tl.load(x_row + cols, mask=None, eviction_policy='evict_first')

    # One-pass mean/variance
    sum_ = tl.sum(x, axis=0)
    mean = sum_ / N
    sum_sq = tl.sum(x * x, axis=0)
    var = sum_sq / N - mean * mean
    var = tl.where(var < 0, 0.0, var)    # guard against numerical negatives
    rstd = tl.rsqrt(var + eps)

    out = (x - mean) * rstd
    # Store with eviction hint: output will be used later, evict last
    tl.store(out_row + cols, out, mask=None, eviction_policy='evict_last')


def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous()
    M, N = x.shape
    out = torch.empty_like(x)

    BLOCK = N  # exactly one block per row
    grid = (M,)

    layer_norm_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        N=N,
        eps=1e-5,
        BLOCK=BLOCK,
        num_warps=4,
        num_stages=1,
    )
    return out