import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # 2 warps: stages 2..4
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=2, num_stages=4),
        # 4 warps: stages 2..5
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=4),
        triton.Config({}, num_warps=4, num_stages=5),
        # 8 warps: stages 2..6
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=4),
        triton.Config({}, num_warps=8, num_stages=5),
        triton.Config({}, num_warps=8, num_stages=6),
        # 16 warps: stages 2..5 (16 warps * 32 = 512 threads, still ok for block size 2048)
        triton.Config({}, num_warps=16, num_stages=2),
        triton.Config({}, num_warps=16, num_stages=3),
        triton.Config({}, num_warps=16, num_stages=4),
        triton.Config({}, num_warps=16, num_stages=5),
    ],
    key=[],
)
@triton.jit
def rms_norm_kernel(x_ptr, y_ptr, N, D: tl.constexpr, eps, BLOCK_SIZE: tl.constexpr):
    """
    RMS normalization without affine weights.
    Each program processes one row. D is a multiple of BLOCK_SIZE.
    """
    # D=4096, BLOCK_SIZE=2048 -> exactly two iterations, no masking needed
    tl.static_assert(D % BLOCK_SIZE == 0)

    row_id = tl.program_id(0)
    if row_id >= N:
        return

    row_start = row_id * D

    # ---------- first pass: sum of squares ----------
    sum_sq = 0.0
    for col_start in range(0, D, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        x_vals = tl.load(x_ptr + row_start + cols)
        x_sq = x_vals * x_vals
        sum_sq += tl.sum(x_sq, axis=0)

    # compute rms coefficient
    mean_sq = sum_sq / D
    rms = tl.rsqrt(mean_sq + eps)

    # ---------- second pass: scale and store ----------
    for col_start in range(0, D, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        x_vals = tl.load(x_ptr + row_start + cols)
        y_vals = x_vals * rms
        tl.store(y_ptr + row_start + cols, y_vals)


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
    rms_norm_kernel[grid](x, y, N, D=D, eps=1e-5, BLOCK_SIZE=2048)
    return y