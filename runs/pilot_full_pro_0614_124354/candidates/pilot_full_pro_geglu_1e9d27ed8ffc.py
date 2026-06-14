import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
import math

@triton.jit
def geglu_kernel(
    x_ptr, y_ptr,
    M, N,
    in_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # row and column indices for the output tile
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]   # (BLOCK_M, 1)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]   # (1, BLOCK_N)

    mask_m = rm < M
    mask_n = rn < N
    mask = mask_m & mask_n

    # load a (left half of input) and b (right half)
    a_idx = rm * in_dim + rn
    b_idx = rm * in_dim + rn + N
    a = tl.load(x_ptr + a_idx, mask=mask, other=0.0)
    b = tl.load(x_ptr + b_idx, mask=mask, other=0.0)

    # GELU with tanh approximation
    sqrt2pi = 0.7978845608028654   # sqrt(2/pi)
    coeff   = 0.044715
    a3      = a * a * a
    inner   = sqrt2pi * (a + coeff * a3)
    gelu    = 0.5 * a * (1.0 + libdevice.tanh(inner))

    out = gelu * b

    # store output
    y_idx = rm * N + rn
    tl.store(y_ptr + y_idx, out, mask=mask)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """GEGLU: (8192, 8192) -> (8192, 4096)"""
    assert x.is_cuda and x.dtype == torch.float32
    assert x.shape == (8192, 8192)

    M, N = 8192, 4096
    out = torch.empty(M, N, dtype=torch.float32, device=x.device)

    # Tuned 2D tile sizes (strategy: tune_block_size)
    BLOCK_M = 16
    BLOCK_N = 256
    num_warps = 8

    grid_m = triton.cdiv(M, BLOCK_M)
    grid_n = triton.cdiv(N, BLOCK_N)
    grid = (grid_m, grid_n)

    geglu_kernel[grid](
        x, out, M, N,
        in_dim=8192,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )
    return out