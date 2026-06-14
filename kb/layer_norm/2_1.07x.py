import torch
import triton
import triton.language as tl

@triton.jit
def layer_norm_kernel(
    x_ptr, out_ptr,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # One program per row
    row = tl.program_id(0)
    x_row_ptr = x_ptr + row * N
    out_row_ptr = out_ptr + row * N

    # Column offsets – full row
    offsets = tl.arange(0, BLOCK_SIZE)
    # N and BLOCK_SIZE are equal, so no mask required
    x = tl.load(x_row_ptr + offsets)

    # Single pass: compute sum and sum of squares
    sum_x = tl.sum(x, axis=0)
    sum_xsq = tl.sum(x * x, axis=0)

    # Compute mean and variance (numerically stable via max with 0)
    mean = sum_x / N
    var = sum_xsq / N - mean * mean
    var = tl.where(var < 0.0, 0.0, var)

    # Standard deviation
    inv_std = tl.rsqrt(var + eps)

    # Normalize
    out = (x - mean) * inv_std
    tl.store(out_row_ptr + offsets, out)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous()
    M, N = x.shape
    out = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(N)  # 4096

    grid = (M,)
    layer_norm_kernel[grid](
        x, out,
        N=N,
        eps=1e-5,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,      # good balance for BLOCK_SIZE=4096 reduction
        num_stages=1,     # no pipelining needed
    )
    return out