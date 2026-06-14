import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # small block – few warps, pipelined
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=3),

        # medium block
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16, num_stages=2),

        # large block – wide vectorization, moderate warps
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16, num_stages=4),

        # extra large block – near row size
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=32, num_stages=2),

        # one-shot block – entire row in one tile
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=32, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=32, num_stages=4),

        # stress pipelining for smaller tiles
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=6),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=6),
    ],
    key=[],
)
@triton.jit
def rms_norm_kernel(x_ptr, y_ptr, N, D: tl.constexpr, eps, BLOCK_SIZE: tl.constexpr):
    """
    RMS normalization without affine weights.
    Each program processes one row in two passes:
      1) compute sum of squares over the row,
      2) scale and write x * rsqrt(mean_sq + eps).
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
        x_vals = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0)
        x_sq = x_vals * x_vals
        sum_sq += tl.sum(x_sq, axis=0)

    # compute rms coefficient
    mean_sq = sum_sq / D
    rms = tl.rsqrt(mean_sq + eps)

    # ---------- second pass: scale and store ----------
    for col_start in range(0, D, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        x_vals = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0)
        y_vals = x_vals * rms
        tl.store(y_ptr + row_start + cols, y_vals, mask=mask)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    Applies RMS normalization to the last dimension.
    Signature: triton_run(x: f32[8192, 4096]) -> f32[8192, 4096]
    """
    N, D = x.shape
    # The operator is defined for this exact shape, but we keep a flexible check.
    assert D == 4096, f"Expected last dim 4096, got {D}"
    x = x.contiguous()
    y = torch.empty_like(x)

    grid = (N,)
    rms_norm_kernel[grid](x, y, N, D, 1e-5)  # D passed as constexpr, eps=1e-5
    return y