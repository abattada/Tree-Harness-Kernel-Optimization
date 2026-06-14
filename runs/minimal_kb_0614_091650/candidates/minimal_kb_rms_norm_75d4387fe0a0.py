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

    # Contiguous columns offsets – no mask since N == BLOCK
    offsets = tl.arange(0, BLOCK)
    # Hint: the offsets are contiguous for the entire block
    offsets = tl.max_contiguous(tl.multiple_of(offsets, BLOCK), BLOCK)

    # Load input row with eviction hint (stream once, no reuse)
    x = tl.load(x_row_ptr + offsets, eviction_policy='evict_first')

    # Compute sum of squares
    x_sq = x * x
    sum_sq = tl.sum(x_sq, axis=0)

    # Use reciprocal of N to avoid division inside kernel
    rcp_N = 1.0 / N
    mean_sq = sum_sq * rcp_N
    rstd = tl.rsqrt(mean_sq + eps)

    out = x * rstd
    tl.store(out_row_ptr + offsets, out, eviction_policy='evict_first')


def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous()
    M, N = x.shape
    out = torch.empty_like(x)

    BLOCK = N  # matches the number of columns exactly
    grid = (M,)

    rms_norm_kernel[grid](
        x, out,
        N=N,
        eps=1e-5,
        BLOCK=BLOCK,
        num_warps=4,      # 4 warps is enough for BLOCK=4096 reduction
        num_stages=1,     # no pipelining needed for this workload
    )
    return out