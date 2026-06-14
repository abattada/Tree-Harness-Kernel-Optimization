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
    x_offsets = tl.arange(0, BLOCK)
    x_base = x_ptr + row * N
    out_base = out_ptr + row * N
    # Since BLOCK == N, no mask needed
    x = tl.load(x_base + x_offsets)
    x_sq = x * x
    sum_sq = tl.sum(x_sq, axis=0)
    mean_sq = sum_sq / N
    rstd = tl.rsqrt(mean_sq + eps)
    out = x * rstd
    tl.store(out_base + x_offsets, out)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous()
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK = N  # 4096, power of two
    grid = (M,)
    rms_norm_kernel[grid](
        x, out,
        N=N,
        eps=1e-5,
        BLOCK=BLOCK,
        num_warps=4,
        num_stages=1,
    )
    return out