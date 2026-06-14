import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    x_ptr,
    out_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute RMSNorm: out = x * rsqrt(mean(x^2, dim=-1) + eps)
    Each program processes one row, looping over BLOCK_SIZE elements at a time.
    """
    row = tl.program_id(0)
    row_start = row * n_cols

    # --- Pass 1: compute sum of squares ---
    sum_sq = 0.0
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + row_start + cols)
        sum_sq += tl.sum(x * x)

    # --- Compute scale = 1 / sqrt(mean + eps) ---
    rsqrt = tl.rsqrt(sum_sq / n_cols + eps)

    # --- Pass 2: scale and store ---
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + row_start + cols)
        tl.store(out_ptr + row_start + cols, x * rsqrt)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    RMS norm without learnable affine parameters.
    Input:  float32 tensor of shape [8192, 4096]
    Output: float32 tensor of shape [8192, 4096]
    """
    # Input shape is known at compile time and always [8192, 4096]
    N, D = x.shape
    out = torch.empty_like(x)

    # Tuning knobs: BLOCK_SIZE, num_warps, num_stages
    BLOCK_SIZE = 1024  # divides 4096 cleanly

    # One program per row, 8192 programs total → good occupancy
    grid = (N,)
    rms_norm_kernel[grid](
        x, out,
        n_cols=D,
        eps=1e-5,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
    )
    return out