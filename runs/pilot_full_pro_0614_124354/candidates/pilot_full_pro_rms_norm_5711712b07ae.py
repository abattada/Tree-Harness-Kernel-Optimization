import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16, num_stages=2),
    ],
    key=[],
)
@triton.jit
def rms_norm_fused_kernel(x_ptr, y_ptr, N, D: tl.constexpr, eps,
                           BLOCK_SIZE: tl.constexpr):
    """
    Fused RMS normalisation kernel.
    Each program processes one row:
      1) Load the row into shared memory while computing sum of squares on the fly.
      2) Compute 1/sqrt(mean_sq + eps) using the accumulated sum.
      3) Scale the row using the shared copy and write the result.
    The entire row is read from DRAM only once.
    """
    row_id = tl.program_id(0)
    if row_id >= N:
        return

    row_start = row_id * D
    tid = tl.arange(0, BLOCK_SIZE)

    # Shared memory holds the whole row
    x_shared = tl.alloc_shared((D,), tl.float32)

    sum_sq = 0.0

    # -- Pass 1: load, store to shared, accumulate sum of squares --
    for off in range(0, D, BLOCK_SIZE):
        cols = off + tid
        mask = cols < D
        val = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0)
        tl.store(x_shared + cols, val, mask=mask)
        sum_sq += tl.sum(val * val, axis=0)

    # Make sure all stores to shared memory are visible
    tl.debug_barrier()

    # Compute the RMS scale factor
    rms = tl.rsqrt(sum_sq / D + eps)

    # -- Pass 2: read from shared, scale, and write output --
    for off in range(0, D, BLOCK_SIZE):
        cols = off + tid
        mask = cols < D
        val = tl.load(x_shared + cols, mask=mask, other=0.0)
        out = val * rms
        tl.store(y_ptr + row_start + cols, out, mask=mask)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    Applies RMS normalization along the last dimension.
    Signature: triton_run(x: f32[8192, 4096]) -> f32[8192, 4096]
    """
    N, D = x.shape
    assert D == 4096, f"Expected last dim 4096, got {D}"
    x = x.contiguous()
    y = torch.empty_like(x)
    grid = (N,)
    rms_norm_fused_kernel[grid](x, y, N, D, 1e-5)
    return y