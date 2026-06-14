import torch
import triton
import triton.language as tl

@triton.jit
def layer_norm_kernel(
    x_ptr, out_ptr,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    x_row_ptr = x_ptr + row * N
    out_row_ptr = out_ptr + row * N

    offsets = tl.arange(0, BLOCK)
    # N == BLOCK, so no mask needed
    x = tl.load(x_row_ptr + offsets, mask=None)

    # One-pass mean and variance
    sum_ = tl.sum(x, axis=0)
    mean = sum_ * (1.0 / N)
    sum_sq = tl.sum(x * x, axis=0)
    var = sum_sq * (1.0 / N) - mean * mean
    # Guard against numerical negatives
    var = tl.where(var < 0, 0.0, var)
    rstd = tl.rsqrt(var + eps)

    out = (x - mean) * rstd
    tl.store(out_row_ptr + offsets, out, mask=None)

def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous()
    M, N = x.shape
    out = torch.empty_like(x)

    BLOCK = N  # N=4096, exactly one block per row
    grid = (M,)
    layer_norm_kernel[grid](
        x, out,
        N=N,
        eps=1e-5,
        BLOCK=BLOCK,
        num_warps=4,      # higher occupancy, fewer registers
        num_stages=1,
    )
    return out