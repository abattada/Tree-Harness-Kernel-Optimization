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
    """Normalize each row: x * rsqrt(mean(x^2) + eps)."""
    row = tl.program_id(0)
    x_row_ptr = x_ptr + row * N
    out_row_ptr = out_ptr + row * N

    offsets = tl.arange(0, BLOCK)
    # Stream input data; no mask needed because N is exact multiple of BLOCK
    x = tl.load(x_row_ptr + offsets, mask=None, eviction_policy='evict_first')

    # Compute RMS in one pass
    x_sq = x * x
    sum_sq = tl.sum(x_sq, axis=0)
    mean_sq = sum_sq / N
    rstd = tl.rsqrt(mean_sq + eps)

    out = x * rstd
    tl.store(out_row_ptr + offsets, out, mask=None, eviction_policy='evict_last')


def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous(), "Input tensor must be contiguous"
    M, N = x.shape
    out = torch.empty_like(x)

    # One tile per row; N must be divisible by BLOCK
    BLOCK = 1024  # A sensible power-of-two tile, exact divisor of 4096
    assert N % BLOCK == 0, "N must be divisible by BLOCK"
    # Number of tiles along the row dimension
    num_tiles = N // BLOCK

    # We will process the row in a grid of (M * num_tiles) programs,
    # each handling a contiguous chunk of BLOCK elements.
    # This gives more programs and potentially higher occupancy.
    grid = (M * num_tiles,)

    rms_norm_kernel[grid](
        x, out,
        N=N,
        eps=1e-5,
        BLOCK=BLOCK,
        num_warps=4,   # Smaller warps per block to increase occupancy
        num_stages=1,
    )
    return out