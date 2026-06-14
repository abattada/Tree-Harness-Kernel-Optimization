import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def geglu_kernel(
    x_ptr,                # pointer to input [M, K] (row-major flattened)
    out_ptr,              # pointer to output [M, N]
    stride_x_row,         # row stride of x in elements (=K)
    stride_out_row,       # row stride of out in elements (=N)
    BLOCK_SIZE: tl.constexpr,  # must equal N (full output row, no mask)
    N: tl.constexpr,           # output dimension = K//2
):
    # Safety: block covers the entire output row → no boundary checks needed
    tl.static_assert(BLOCK_SIZE == N, "BLOCK_SIZE must equal N")

    row = tl.program_id(0)

    col_offs = tl.arange(0, BLOCK_SIZE)

    x_row = x_ptr + row * stride_x_row
    out_row = out_ptr + row * stride_out_row

    # Load a (first half) and b (second half) – one-shot, streaming recommended
    a = tl.load(x_row + col_offs, eviction_policy='evict_first')
    b = tl.load(x_row + col_offs + N, eviction_policy='evict_first')

    # GELU with tanh approximation
    sqrt2_over_pi = 0.7978845608028654
    c = 0.044715

    a3 = a * a * a
    inner = sqrt2_over_pi * (a + c * a3)
    tanh_val = libdevice.tanh(inner)
    gelu_a = 0.5 * a * (1.0 + tanh_val)

    out = gelu_a * b

    # Store result (streaming write)
    tl.store(out_row + col_offs, out, eviction_policy='evict_first')


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    x: float32 tensor of shape [M, K] with K even (here [8192, 8192])
    returns: float32 tensor of shape [M, K//2] = [8192, 4096]
    """
    M, K = x.shape
    N = K // 2                     # 4096

    out = torch.empty(M, N, device=x.device, dtype=torch.float32)

    BLOCK_SIZE = N                 # full row

    grid = (M,)                    # one program per row

    geglu_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        N=N,
        num_warps=8,               # lower warp count reduces register pressure
        num_stages=4,              # increase pipelining to hide load latency
    )

    return out