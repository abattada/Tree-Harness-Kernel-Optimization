import torch
import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(
    x_ptr,
    out_ptr,
    stride_x,
    stride_out,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    x_row_ptr = x_ptr + row * stride_x
    out_row_ptr = out_ptr + row * stride_out

    offsets = tl.arange(0, BLOCK)
    # N == BLOCK for our shape, but we keep mask for correctness if N < BLOCK
    mask = offsets < N
    x = tl.load(x_row_ptr + offsets, mask=mask)
    x_sq = x * x
    sum_sq = tl.sum(x_sq, axis=0)
    mean_sq = sum_sq / N
    rms = tl.rsqrt(mean_sq + eps)
    out = x * rms
    tl.store(out_row_ptr + offsets, out, mask=mask)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous()
    M, N = x.shape
    out = torch.empty_like(x)

    # BLOCK_SIZE = nearest power of two >= N (ensures coalescing)
    BLOCK = triton.next_power_of_2(N)
    eps = 1e-5

    grid = (M,)
    rms_norm_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        N=N,
        eps=eps,
        BLOCK=BLOCK,
        num_warps=4,      # empirically good for 4096-wide rows
        num_stages=1,     # no pipelining needed for this simple kernel
    )
    return out