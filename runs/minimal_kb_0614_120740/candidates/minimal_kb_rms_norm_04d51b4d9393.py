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
    """
    One program per row. BLOCK = N.
    """
    row = tl.program_id(0)
    x_row_ptr = x_ptr + row * N
    out_row_ptr = out_ptr + row * N

    offsets = tl.arange(0, BLOCK)
    # Load entire row – evict first because each element is used only once
    x = tl.load(x_row_ptr + offsets, mask=None, eviction_policy='evict_first')

    # Compute RMS
    x_sq = x * x
    sum_sq = tl.sum(x_sq, axis=0)
    mean_sq = sum_sq / N
    rstd = tl.rsqrt(mean_sq + eps)

    out = x * rstd
    # Store – hint that the output may be reused later
    tl.store(out_row_ptr + offsets, out, mask=None, eviction_policy='evict_last')


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    x: (8192, 4096) float32, contiguous.
    Returns: same shape, rms-normalized.
    """
    assert x.is_contiguous()
    M, N = x.shape
    out = torch.empty_like(x)

    # BLOCK = N (4096), exactly one block per row, no masking needed
    BLOCK = N

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