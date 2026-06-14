import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
import math


@triton.jit
def geglu_kernel(x_ptr, out_ptr, N_COLS: tl.constexpr, BLOCK: tl.constexpr):
    # Grid: (row, col_block)
    row = tl.program_id(0)
    col_block = tl.program_id(1)

    # Element offsets for this block in the output (and first half of input)
    offs = col_block * BLOCK + tl.arange(0, BLOCK)

    # Coalesced loads of the two halves along the last dimension
    a = tl.load(x_ptr + row * N_COLS + offs)
    b = tl.load(x_ptr + row * N_COLS + offs + N_COLS // 2)

    # GELU with tanh approximation
    sqrt_2_pi = 0.7978845608028654
    coeff = 0.044715
    x_ = a
    x3 = x_ * x_ * x_
    arg = sqrt_2_pi * (x_ + coeff * x3)
    tanh_val = libdevice.tanh(arg)
    gelu_val = 0.5 * x_ * (1.0 + tanh_val)

    out = gelu_val * b
    tl.store(out_ptr + row * (N_COLS // 2) + offs, out)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    N, C = x.shape
    # Allocate output: half the last dimension
    out = torch.empty(N, C // 2, dtype=x.dtype, device=x.device)

    # Power-of-two block size that exactly divides 4096 for the provided shape
    BLOCK = 1024
    grid = (N, (C // 2) // BLOCK)

    geglu_kernel[grid](x, out, N_COLS=C, BLOCK=BLOCK)
    return out