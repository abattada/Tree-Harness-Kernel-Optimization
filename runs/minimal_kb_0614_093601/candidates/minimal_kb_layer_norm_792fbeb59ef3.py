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
    row = tl.program_id(0)
    x_row_ptr = x_ptr + row * stride_x
    out_row_ptr = out_ptr + row * stride_out

    offsets = tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(x_row_ptr + offsets, mask=mask)

    # Compute mean and variance in one pass
    sum_ = tl.sum(x, axis=0)
    mean = sum_ / N
    sum_sq = tl.sum(x * x, axis=0)
    var = sum_sq / N - mean * mean
    # Guard against numerical negatives
    var = tl.where(var < 0, 0.0, var)
    rstd = tl.rsqrt(var + eps)

    out = (x - mean) * rstd
    tl.store(out_row_ptr + offsets, out, mask=mask)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous()
    M, N = x.shape
    out = torch.empty_like(x)

    BLOCK = triton.next_power_of_2(N)  # 4096 in this case
    grid = (M,)
    layer_norm_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        N=N,
        eps=1e-5,
        BLOCK=BLOCK,
        num_warps=8,
        num_stages=1,
    )
    return out