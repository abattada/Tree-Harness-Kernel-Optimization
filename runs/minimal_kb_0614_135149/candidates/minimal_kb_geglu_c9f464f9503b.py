import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def geglu_kernel(
    x_ptr,                        # input [M, K] (flattened)
    out_ptr,                      # output [M, N] (flattened)
    stride_x_row,                 # row stride of x in elements (K)
    stride_out_row,               # row stride of out in elements (N)
    N: tl.constexpr,              # half the inner dimension (must be a power of two)
    BLOCK_SIZE: tl.constexpr,     # tile width, divides N evenly
):
    # Row index: one program per row and tile column
    row = tl.program_id(0)
    col_block = tl.program_id(1)

    # Start column for this tile.  BLOCK_SIZE divides N → no boundary mask needed.
    col_start = col_block * BLOCK_SIZE
    col_offs = col_start + tl.arange(0, BLOCK_SIZE)

    # Base pointers for this row
    x_row = x_ptr + row * stride_x_row
    out_row = out_ptr + row * stride_out_row

    # ---- load a (first half) and b (second half) --------------------------
    a = tl.load(x_row + col_offs)               # tile of a
    b = tl.load(x_row + col_offs + N)           # corresponding tile of b

    # ---- GELU with tanh approximation -------------------------------------
    # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt2_over_pi = 0.7978845608028654
    c = 0.044715

    a3 = a * a * a
    inner = sqrt2_over_pi * (a + c * a3)
    gelu_a = 0.5 * a * (1.0 + libdevice.tanh(inner))

    # ---- gate and store ---------------------------------------------------
    out = gelu_a * b
    tl.store(out_row + col_offs, out)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    x : float32 tensor of shape [8192, 8192]
    returns : float32 tensor of shape [8192, 4096]
    """
    M, K = x.shape
    N = K // 2                          # output width = 4096

    # Allocate output
    out = torch.empty(M, N, device=x.device, dtype=torch.float32)

    # Tile size that divides N evenly – good balance between register pressure
    # and memory latency hiding.  N = 4096, so 1024 works perfectly.
    BLOCK_SIZE = 1024
    grid = (M, N // BLOCK_SIZE)         # 8192 blocks along rows, 4 blocks along columns

    geglu_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,                    # smaller tile → fewer threads → lower register usage
        num_stages=3,                   # moderate pipelining
    )

    return out