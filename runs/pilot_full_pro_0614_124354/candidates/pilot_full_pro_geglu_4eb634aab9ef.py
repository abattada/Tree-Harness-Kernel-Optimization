import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def geglu_kernel_vec(
    x_ptr,
    y_ptr,
    out_dim: tl.constexpr,   # 4096
    in_dim: tl.constexpr,    # 8192
):
    row = tl.program_id(0)
    # Full output row (no mask needed because out_dim divides exactly)
    offs = tl.arange(0, out_dim)

    # Left half: a
    a_offs = row * in_dim + offs
    a = tl.load(x_ptr + a_offs)

    # Right half: b
    b_offs = row * in_dim + out_dim + offs
    b = tl.load(x_ptr + b_offs)

    # GELU with tanh approximation (identical to PyTorch's approximate='tanh')
    sqrt2pi = 0.7978845608028654   # sqrt(2/pi)
    coeff = 0.044715
    a3 = a * a * a
    inner = sqrt2pi * (a + coeff * a3)
    gelu = 0.5 * a * (1.0 + libdevice.tanh(inner))

    out = gelu * b

    # Store contiguous output row
    y_offs = row * out_dim + offs
    tl.store(y_ptr + y_offs, out)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """GEGLU: x.shape = (8192, 8192) -> output shape (8192, 4096)"""
    assert x.is_cuda and x.dtype == torch.float32
    assert x.shape == (8192, 8192)

    out = torch.empty(8192, 4096, dtype=torch.float32, device=x.device)
    rows = 8192

    # Launch one program per row – each processes contiguous chunks of length 4096
    grid = (rows,)
    geglu_kernel_vec[grid](
        x, out,
        out_dim=4096, in_dim=8192,
        num_warps=8, num_stages=3,
    )
    return out