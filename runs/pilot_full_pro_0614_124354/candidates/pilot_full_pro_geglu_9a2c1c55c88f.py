import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
import math

@triton.jit
def geglu_kernel(
    x_ptr, y_ptr,
    N,
    out_dim: tl.constexpr,
    in_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # row, col indices for output (= position in half x)
    row = offs // out_dim
    col = offs % out_dim

    # load a (left half) and b (right half)
    a_idx = row * in_dim + col
    b_idx = row * in_dim + col + out_dim
    a = tl.load(x_ptr + a_idx, mask=mask, other=0.0)
    b = tl.load(x_ptr + b_idx, mask=mask, other=0.0)

    # gelu_tanh(a)
    sqrt2pi = 0.7978845608028654      # sqrt(2/pi)
    coeff = 0.044715
    a3 = a * a * a
    inner = sqrt2pi * (a + coeff * a3)
    gelu = 0.5 * a * (1.0 + libdevice.tanh(inner))

    out = gelu * b
    tl.store(y_ptr + offs, out, mask=mask)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """GEGLU: x.shape = (8192, 8192) -> output shape (8192, 4096)"""
    assert x.is_cuda and x.dtype == torch.float32
    assert x.shape == (8192, 8192)

    out = torch.empty(8192, 4096, dtype=torch.float32, device=x.device)
    N = 8192 * 4096          # total output elements
    out_dim = 4096           # columns in output (= half)
    in_dim = 8192            # original input columns

    # Tuned parameters: larger block size and more warps for better occupancy
    BLOCK_SIZE = 4096
    num_warps = 16
    num_stages = 4

    grid = (triton.cdiv(N, BLOCK_SIZE),)
    geglu_kernel[grid](
        x, out, N,
        out_dim, in_dim, BLOCK_SIZE,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out