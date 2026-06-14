import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    x_ptr,
    out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    VEC_SIZE: tl.constexpr,
    eps: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * N
    row_start = tl.multiple_of(row_start, N)  # hint

    # Static assertion: no boundary mask needed for this shape
    tl.static_assert(N % (BLOCK_SIZE * VEC_SIZE) == 0)

    # Precompute 1/N for fast multiply instead of division
    inv_N = 1.0 / N

    # First pass: compute sum of squares
    total_sq = 0.0  # scalar accumulator
    for tile_start in range(0, N, BLOCK_SIZE * VEC_SIZE):
        offs = (
            tile_start
            + tl.arange(0, BLOCK_SIZE)[:, None] * VEC_SIZE
            + tl.arange(0, VEC_SIZE)[None, :]
        )
        x = tl.load(x_ptr + row_start + offs)
        total_sq += tl.sum(x * x)

    mean_sq = total_sq * inv_N       # fast multiply replaces division
    inv_rms = tl.rsqrt(mean_sq + eps)

    # Second pass: rescale and store
    for tile_start in range(0, N, BLOCK_SIZE * VEC_SIZE):
        offs = (
            tile_start
            + tl.arange(0, BLOCK_SIZE)[:, None] * VEC_SIZE
            + tl.arange(0, VEC_SIZE)[None, :]
        )
        x = tl.load(x_ptr + row_start + offs)
        out = x * inv_rms
        tl.store(out_ptr + row_start + offs, out)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """RMSNorm: x * rsqrt(mean(x^2, dim=-1, keepdim=True) + 1e-5)"""
    M, N = x.shape
    assert (M, N) == (8192, 4096), "Only shape [8192, 4096] is supported"
    out = torch.empty_like(x)

    BLOCK_SIZE = 256
    VEC_SIZE = 4
    eps = 1e-5
    grid = (M,)

    rms_norm_kernel[grid](
        x,
        out,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
        VEC_SIZE=VEC_SIZE,
        eps=eps,
        num_warps=8,
    )
    return out