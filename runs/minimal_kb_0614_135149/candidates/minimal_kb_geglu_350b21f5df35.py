import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def geglu_kernel(
    x_ptr,              # pointer to input [M, K] (flattened)
    out_ptr,            # pointer to output [M, N] (flattened)
    stride_x,           # row stride of x (elements) = K
    stride_out,         # row stride of out (elements) = N
    BLOCK_SIZE: tl.constexpr,   # = N = K//2
):
    # Full-row specialisation: every block covers exactly one output row.
    tl.static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a multiple of 32")

    # row index
    row = tl.program_id(0)

    # pointers to the start of this row
    x_row  = x_ptr  + row * stride_x
    out_row = out_ptr + row * stride_out

    # aligned contiguous offsets (full row, no mask)
    col_offs = tl.arange(0, BLOCK_SIZE)

    # Load a (first half) and b (second half) – streaming, evict after use.
    # Hints: align to 16 bytes (4 floats) to enable wide vectorised loads.
    a = tl.load(x_row + col_offs,
                eviction_policy='evict_first')
    b = tl.load(x_row + col_offs + BLOCK_SIZE,
                eviction_policy='evict_first')

    # GELU with tanh approximation
    sqrt2_over_pi = 0.7978845608028654
    c = 0.044715

    a3 = a * a * a
    inner = sqrt2_over_pi * (a + c * a3)
    tanh_val = libdevice.tanh(inner)
    gelu_a = 0.5 * a * (1.0 + tanh_val)

    # Gating
    out = gelu_a * b

    # Store result (streaming write)
    tl.store(out_row + col_offs, out,
             eviction_policy='evict_first')


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    x: torch.Tensor of shape [8192, 8192] (float32)
    returns: torch.Tensor of shape [8192, 4096] (float32)
    """
    M, K = x.shape
    N = K // 2                     # 4096

    out = torch.empty(M, N, device=x.device, dtype=torch.float32)

    BLOCK_SIZE = N                 # covers the whole output row

    # Launch configuration: more warps to keep every SM occupied,
    # fewer pipeline stages to save registers for the wider block.
    grid = (M,)
    geglu_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=16,
        num_stages=2,
    )
    return out