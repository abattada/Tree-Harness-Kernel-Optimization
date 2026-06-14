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
    # One program per row: each block handles the entire row of N = 4096 columns.
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)  # [0, 1, ..., BLOCK-1]

    # Load the entire row with streaming hint.
    x = tl.load(x_ptr + row * N + offsets, mask=None, eviction_policy='evict_first')

    # Compute RMS = sqrt(mean(x^2) + eps).
    x_sq = x * x
    sum_sq = tl.sum(x_sq, axis=0)      # parallel reduction within the block
    mean_sq = sum_sq / N                # N is a constexpr, compiler can optimise
    rstd = tl.rsqrt(mean_sq + eps)

    out = x * rstd
    # Store the normalised row with hint that the result may be reused later.
    tl.store(out_ptr + row * N + offsets, out, mask=None, eviction_policy='evict_last')


def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous()
    M, N = x.shape  # 8192, 4096
    out = torch.empty_like(x)

    # One thread block per row; BLOCK == N, so no masking needed.
    grid = (M,)

    rms_norm_kernel[grid](
        x, out,
        N=N,
        eps=1e-5,
        BLOCK=N,
        num_warps=8,
        num_stages=1,
    )
    return out