import torch
import triton
import triton.language as tl

# Triton kernel for RMS Normalization (no affine, specific to 8192x4096 shape)
# N=4096, BLOCK_SIZE=4096 (exact fit, no masking needed)
@triton.jit
def rms_norm_kernel(
    x_ptr, out_ptr,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_row_ptr = x_ptr + row * N
    out_row_ptr = out_ptr + row * N

    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_row_ptr + cols)
    x_sq = x * x
    sum_sq = tl.sum(x_sq, axis=0)
    mean_sq = sum_sq / N
    rstd = tl.rsqrt(mean_sq + eps)
    out = x * rstd
    tl.store(out_row_ptr + cols, out)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous()
    M, N = x.shape
    out = torch.empty_like(x)
    # BLOCK_SIZE matches N exactly (power of two)
    BLOCK_SIZE = triton.next_power_of_2(N)  # 4096
    grid = (M,)
    rms_norm_kernel[grid](
        x, out,
        N=N,
        eps=1e-5,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=1,
    )
    return out