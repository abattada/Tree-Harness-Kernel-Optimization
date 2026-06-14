import torch
import triton
import triton.language as tl

# Define constants for RMS norm kernel
@triton.jit
def _rms_norm_kernel(
    x_ptr,
    out_ptr,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Program ID along rows
    row = tl.program_id(0)
    # Base pointer for this row
    x_row_ptr = x_ptr + row * N
    out_row_ptr = out_ptr + row * N

    # Block of columns
    offsets = tl.arange(0, BLOCK)
    # Since N == BLOCK, no masking needed
    x = tl.load(x_row_ptr + offsets, mask=None)
    # Compute square and sum
    x_sq = x * x
    sum_sq = tl.sum(x_sq, axis=0)
    # RMS norm statistics
    mean_sq = sum_sq / N
    rms = tl.rsqrt(mean_sq + eps)
    # Normalize
    out = x * rms
    tl.store(out_row_ptr + offsets, out, mask=None)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous()
    M, N = x.shape
    # Kernel launch configuration
    BLOCK = N  # exactly one block per row
    eps = 1e-5
    out = torch.empty_like(x)

    # Grid: one program per row
    grid = (M,)
    _rms_norm_kernel[grid](
        x,
        out,
        N=N,
        eps=eps,
        BLOCK=BLOCK,
        num_warps=4,
    )
    return out