import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ---------------------------------------------------------------------------
# GeGLU kernel – gated GELU(tanh approx)
# One program per row, full-row block, no masks.
# Increased num_warps to 16 to reduce per-thread work and improve occupancy.
# ---------------------------------------------------------------------------

@triton.jit
def geglu_kernel(
    x_ptr,                     # input [M, K] flat
    out_ptr,                   # output [M, K//2] flat
    stride_x: tl.constexpr,    # row stride of x in elements
    stride_out: tl.constexpr,  # row stride of out in elements
    BLOCK_SIZE: tl.constexpr,  # == K//2, covers whole output row
):
    row = tl.program_id(0)

    x_row = x_ptr + row * stride_x
    out_row = out_ptr + row * stride_out

    col_offs = tl.arange(0, BLOCK_SIZE)

    a = tl.load(x_row + col_offs, eviction_policy='evict_first')
    b = tl.load(x_row + col_offs + BLOCK_SIZE, eviction_policy='evict_first')

    # GELU(x) = 0.5 * x * (1 + tanh( sqrt(2/π) * (x + 0.044715·x³) ))
    sqrt2_over_pi = 0.7978845608028654
    c = 0.044715
    a3 = a * a * a
    inner = sqrt2_over_pi * (a + c * a3)
    tanh_val = libdevice.tanh(inner)
    gelu_a = 0.5 * a * (1.0 + tanh_val)

    out = gelu_a * b

    tl.store(out_row + col_offs, out)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    M, K = x.shape
    N = K // 2                     # 4096

    out = torch.empty(M, N, device=x.device, dtype=x.dtype)

    BLOCK_SIZE = N
    grid = (M,)

    geglu_kernel[grid](
        x, out,
        stride_x=x.stride(0),
        stride_out=out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=16,              # increased to improve occupancy
        num_stages=4,              # kept for moderate pipelining
    )
    return out