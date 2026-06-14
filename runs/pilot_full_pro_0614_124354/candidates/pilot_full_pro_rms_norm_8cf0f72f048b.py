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
    row_start = tl.multiple_of(row_start, N)  # hint for the compiler

    # No boundary check needed because shapes divide evenly
    tl.static_assert(N % (BLOCK_SIZE * VEC_SIZE) == 0)

    # Pre‑compute reciprocal to turn division into multiplication (fast_math)
    inv_N: tl.constexpr = 1.0 / N

    # First pass: sum of squares
    total_sq = 0.0
    for tile_start in range(0, N, BLOCK_SIZE * VEC_SIZE):
        offs = (
            tile_start
            + tl.arange(0, BLOCK_SIZE)[:, None] * VEC_SIZE
            + tl.arange(0, VEC_SIZE)[None, :]
        )
        x = tl.load(x_ptr + row_start + offs)
        total_sq += tl.sum(x * x)

    # mean(x^2) = total_sq / N → total_sq * inv_N
    mean_sq = total_sq * inv_N
    # rsqrt replaces 1 / sqrt (fast_math primitive)
    inv_rms = tl.rsqrt(mean_sq + eps)

    # Second pass: rescale and write back
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
    """RMSNorm: x * rsqrt(mean(x^2, dim=-1) + 1e-5) — fast_math variant"""
    M, N = x.shape
    assert (M, N) == (8192, 4096), "Only shape [8192, 4096] is supported"
    out = torch.empty_like(x)

    # Tuned for Blackwell (SM 120), high occupancy
    BLOCK_SIZE = 256   # threads per block (8 warps)
    VEC_SIZE = 4       # contiguous elements per thread
    eps: tl.constexpr = 1e-5
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