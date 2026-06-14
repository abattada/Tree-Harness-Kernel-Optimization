import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def geglu_kernel(
    x_ptr, y_ptr,
    stride_x_row, stride_y_row,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col_block = tl.program_id(1)
    col_offsets = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < 4096

    # Base pointers for the row
    x_base = x_ptr + row * stride_x_row
    y_base = y_ptr + row * stride_y_row

    # Load a (first half) and b (second half)
    a = tl.load(x_base + col_offsets, mask=mask)
    b = tl.load(x_base + col_offsets + 4096, mask=mask)

    # Compute GELU with tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    x3 = a * a * a
    inner = sqrt2_over_pi * (a + 0.044715 * x3)
    tanh_val = libdevice.tanh(inner)
    gelu_a = 0.5 * a * (1.0 + tanh_val)

    result = gelu_a * b
    tl.store(y_base + col_offsets, result, mask=mask)

def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32
    assert x.shape == (8192, 8192)

    y = torch.empty(8192, 4096, dtype=torch.float32, device=x.device)
    BLOCK_SIZE = 1024  # 4096 / 4 = 1024, fine granularity

    grid = (8192, 4096 // BLOCK_SIZE)
    geglu_kernel[grid](
        x, y,
        x.stride(0), y.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=4,
    )
    return y