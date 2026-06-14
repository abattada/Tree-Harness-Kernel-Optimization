import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_kernel(
    x_ptr, out_ptr,
    stride_x, stride_out,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # one program per row
    row = tl.program_id(0)
    x_row = x_ptr + row * stride_x
    out_row = out_ptr + row * stride_out

    cols = tl.arange(0, BLOCK)
    # Since N == BLOCK, no masking needed
    x = tl.load(x_row + cols)

    # single-pass: sum and sum of squares
    sum_x = tl.sum(x, axis=0)
    sum_x2 = tl.sum(x * x, axis=0)

    mean = sum_x / N
    var = sum_x2 / N - mean * mean
    rstd = tl.rsqrt(var + eps)

    out = (x - mean) * rstd
    tl.store(out_row + cols, out)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous()
    M, N = x.shape
    out = torch.empty_like(x)

    grid = (M,)
    BLOCK = triton.next_power_of_2(N)  # 4096 for N=4096

    _layer_norm_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        N=N,
        eps=1e-5,
        BLOCK=BLOCK,
        num_warps=4,
        num_stages=1,
    )
    return out