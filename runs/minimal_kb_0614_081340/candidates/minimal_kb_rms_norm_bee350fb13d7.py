import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    x_ptr,
    out_ptr,
    N: int,
    eps: float,
    BLOCK_SIZE: tl.constexpr,
):
    """
    One program per row.
    Two-pass RMS normalization: first compute sum(x^2), then normalize.
    """
    row = tl.program_id(0)
    base = row * N

    # ---------- pass 1: accumulate sum of squares ----------
    sum_sq = tl.zeros([], tl.float32)
    for start in range(0, N, BLOCK_SIZE):
        off = start + tl.arange(0, BLOCK_SIZE)
        mask = off < N
        x = tl.load(x_ptr + base + off, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x)

    # ---------- pass 2: scale and store ----------
    rms = tl.sqrt(sum_sq / N + eps)
    for start in range(0, N, BLOCK_SIZE):
        off = start + tl.arange(0, BLOCK_SIZE)
        mask = off < N
        x = tl.load(x_ptr + base + off, mask=mask, other=0.0)
        tl.store(out_ptr + base + off, x / rms, mask=mask)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    RMS Normalization (no affine).
    x: f32[8192, 4096]
    returns: f32[8192, 4096]  (x * rsqrt(mean(x^2, -1) + 1e-5))
    """
    M, N = x.shape
    out = torch.empty_like(x)

    # Tuning knobs (seed defaults):
    #   BLOCK_SIZE  – tile along the row direction (power of 2)
    #   num_warps   – threads per warp = 32, total = 32 * num_warps
    #   num_stages  – software pipeline depth
    BLOCK_SIZE = 4096   # single tile per pass → exactly 2 global loads per element
    num_warps = 4       # 128 threads per program
    num_stages = 2      # minimal pipeline

    grid = (M,)
    rms_norm_kernel[grid](
        x, out, N, 1e-5,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out