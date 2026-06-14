import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def geglu_kernel(
    x_ptr,
    out_ptr,
    stride_x,
    stride_out,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Row index
    row = tl.program_id(0)

    # Pointers to rows
    x_row = x_ptr + row * stride_x
    out_row = out_ptr + row * stride_out

    # Column indices (full block, no mask needed because BLOCK_SIZE == N)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # Load a (first half) and b (second half)
    a = tl.load(x_row + col_offsets, eviction_policy='evict_first')
    b = tl.load(x_row + col_offsets + N, eviction_policy='evict_first')

    # GeLU (tanh approximation): 0.5 * a * (1 + tanh(sqrt(2/pi) * (a + 0.044715 * a^3)))
    sqrt2_over_pi = 0.7978845608028654
    a_cube = a * a * a
    inner = sqrt2_over_pi * (a + 0.044715 * a_cube)
    tanh_val = libdevice.tanh(inner)
    gelu_a = 0.5 * a * (1.0 + tanh_val)

    # Gating: out = gelu(a) * b
    out = gelu_a * b

    tl.store(out_row + col_offsets, out, eviction_policy='evict_last')

def triton_run(x: torch.Tensor) -> torch.Tensor:
    M, K = x.shape
    N = K // 2  # output feature dimension
    out = torch.empty(M, N, device=x.device, dtype=torch.float32)

    BLOCK_SIZE = N  # fully cover each row
    grid = (M,)

    geglu_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=16,
        num_stages=4,
    )
    return out