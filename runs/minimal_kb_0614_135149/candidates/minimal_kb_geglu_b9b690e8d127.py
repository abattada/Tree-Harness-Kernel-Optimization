import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def geglu_kernel(
    x_ptr,                # pointer to input [M, K] (flattened)
    out_ptr,              # pointer to output [M, N] (flattened)
    stride_x_row,         # row stride of x in elements (K)
    stride_out_row,       # row stride of out in elements (N)
    BLOCK_SIZE: tl.constexpr,   # equals N (4096)
    N: tl.constexpr,            # output dimension = K // 2
):
    # Safety: block covers the entire output row → no partial masking
    tl.static_assert(BLOCK_SIZE == N, "BLOCK_SIZE must equal N")

    # Row index
    row = tl.program_id(0)

    # Column offsets – always valid, full row
    col_offs = tl.arange(0, BLOCK_SIZE)

    # Compiler hint: row alignment is a multiple of 128 bytes (32 floats).
    # This helps the compiler use wider/more efficient load instructions.
    col_offs_aligned = tl.multiple_of(col_offs, 32)

    # Base pointers for this row
    x_row  = x_ptr  + row * stride_x_row
    out_row = out_ptr + row * stride_out_row

    # Load a (first half) and b (second half) with streaming hints
    a = tl.load(x_row + col_offs_aligned,     eviction_policy='evict_first')
    b = tl.load(x_row + col_offs_aligned + N, eviction_policy='evict_first')

    # GELU with tanh approximation:
    #   GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt2_over_pi = 0.7978845608028654
    c = 0.044715

    a3    = a * a * a
    inner = sqrt2_over_pi * (a + c * a3)
    tanh_val = libdevice.tanh(inner)
    gelu_a = 0.5 * a * (1.0 + tanh_val)

    # Gating
    out = gelu_a * b

    # Store result – streaming again (the output will be consumed later)
    tl.store(out_row + col_offs_aligned, out, eviction_policy='evict_first')


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    x: float32 tensor of shape [8192, 8192]
    returns: float32 tensor of shape [8192, 4096]
    """
    M, K = x.shape
    N = K // 2                     # 4096

    # Allocate output tensor
    out = torch.empty(M, N, device=x.device, dtype=torch.float32)

    BLOCK_SIZE = N                 # exactly one full output row per program

    # One program per row
    grid = (M,)

    geglu_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        N=N,
        num_warps=8,
        num_stages=4,
    )

    return out