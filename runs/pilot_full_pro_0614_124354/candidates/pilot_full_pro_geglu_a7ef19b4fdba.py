import torch
import triton
import triton.language as tl
import math
from triton.language.extra import libdevice


@triton.jit
def geglu_kernel(
    x_ptr, y_ptr,
    M, N, N_out,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused GEGLU: a, b = x.chunk(2, dim=-1); gelu_tanh(a) * b
    Grid is 2D: (M, ceil(N_out / BLOCK_SIZE))
    Each program loads a contiguous chunk of a row, computes gelu(a)*b, and stores.
    """
    pid_m = tl.program_id(0)          # row index
    pid_n = tl.program_id(1)          # column-block index

    row = pid_m
    # column offsets for output (and for a/b)
    col_offs = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offs < N_out

    # a comes from the first half, b from the second half
    a_offs = row * N + col_offs
    b_offs = row * N + col_offs + N_out

    a = tl.load(x_ptr + a_offs, mask=mask, other=0.0)
    b = tl.load(x_ptr + b_offs, mask=mask, other=0.0)

    # GELU approximation: 0.5 * a * (1 + tanh( sqrt(2/pi) * (a + 0.044715*a^3) ))
    sqrt_2_over_pi = 0.7978845608028654  # math.sqrt(2 / math.pi)
    inner = sqrt_2_over_pi * (a + 0.044715 * a * a * a)
    tanh_inner = libdevice.tanh(inner)
    gelu_a = 0.5 * a * (1.0 + tanh_inner)

    result = gelu_a * b
    tl.store(y_ptr + row * N_out + col_offs, result, mask=mask)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    Computes GEGLU (approximate tanh GELU) on the input tensor.
    Input shape: (8192, 8192), output shape: (8192, 4096)
    """
    assert x.ndim == 2 and x.shape[1] % 2 == 0, "Input must be an even-wide 2D tensor"
    M, N = x.shape
    N_out = N // 2

    y = torch.empty((M, N_out), device=x.device, dtype=x.dtype)

    BLOCK_SIZE = 512
    grid = (M, triton.cdiv(N_out, BLOCK_SIZE))

    geglu_kernel[grid](
        x, y,
        M, N, N_out,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
    )
    return y