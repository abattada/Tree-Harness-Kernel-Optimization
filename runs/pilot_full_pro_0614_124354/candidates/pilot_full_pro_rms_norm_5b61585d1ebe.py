import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
    ],
    key=[],
)
@triton.jit
def rms_norm_kernel(x_ptr, y_ptr, N, D: tl.constexpr, eps, BLOCK_SIZE: tl.constexpr):
    """
    RMS normalization without affine weights.
    Uses cache eviction hints:
      - first pass loads are marked 'evict_last' to keep row data in cache
        for the upcoming second pass,
      - second pass loads and the output store are marked 'evict_first'
        because they are not reused within this kernel.
    """
    row_id = tl.program_id(0)
    if row_id >= N:
        return

    row_start = row_id * D

    # ---------- first pass: sum of squares ----------
    sum_sq = 0.0
    for col_start in range(0, D, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        x_vals = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0,
                         eviction_policy='evict_last')
        x_sq = x_vals * x_vals
        sum_sq += tl.sum(x_sq, axis=0)

    # compute rms coefficient
    mean_sq = sum_sq / D
    rms = tl.rsqrt(mean_sq + eps)

    # ---------- second pass: scale and store ----------
    for col_start in range(0, D, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        x_vals = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0,
                         eviction_policy='evict_first')
        y_vals = x_vals * rms
        tl.store(y_ptr + row_start + cols, y_vals, mask=mask,
                 eviction_policy='evict_first')


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    Applies RMS normalization to the last dimension.
    Signature: triton_run(x: f32[8192, 4096]) -> f32[8192, 4096]
    """
    N, D = x.shape
    assert D == 4096, f"Expected last dim 4096, got {D}"
    x = x.contiguous()
    y = torch.empty_like(x)

    grid = (N,)
    rms_norm_kernel[grid](x, y, N, D, 1e-5)  # D passed as constexpr, eps=1e-5
    return y