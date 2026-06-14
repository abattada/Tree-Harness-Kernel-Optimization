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

    offsets = tl.arange(0, BLOCK)

    # Pointers with alignment and contiguity hints for vectorized access
    x_ptrs = x_row_ptr + offsets
    x_ptrs = tl.multiple_of(x_ptrs, 16)       # hint: aligned to 16 fp32 elements
    x_ptrs = tl.max_contiguous(x_ptrs, BLOCK)  # hint: fully contiguous load
    x = tl.load(x_ptrs, mask=None, eviction_policy='evict_first')

    # Single-pass RMS
    x_sq = x * x
    sum_sq = tl.sum(x_sq, axis=0)
    mean_sq = sum_sq / N
    rstd = tl.rsqrt(mean_sq + eps)

    out = x * rstd

    # Store with alignment and contiguity hints
    out_ptrs = out_row_ptr + offsets
    out_ptrs = tl.multiple_of(out_ptrs, 16)
    out_ptrs = tl.max_contiguous(out_ptrs, BLOCK)
    tl.store(out_ptrs, out, mask=None, eviction_policy='evict_last')


def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous()
    M, N = x.shape
    out = torch.empty_like(x)

    BLOCK = N  # each row fits exactly in one tile
    grid = (M,)

    rms_norm_kernel[grid](
        x, out,
        N=N,
        eps=1e-5,
        BLOCK=BLOCK,
        num_warps=8,
        num_stages=1,
    )
    return out