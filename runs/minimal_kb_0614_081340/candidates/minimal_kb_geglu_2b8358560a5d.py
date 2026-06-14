import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# GELU with tanh approximation constant
GELU_COEFF = 0.7978845608028654  # sqrt(2/pi)
GELU_CONST = 0.044715

@triton.jit
def geglu_kernel(
    x_ptr, out_ptr,
    M, N,
    stride_x_m, stride_x_n,
    stride_out_m, stride_out_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    OFFSET_B: tl.constexpr,  # = N (4096)
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile start indices
    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N

    # Compute 2D offsets
    row_offs = row_start + tl.arange(0, BLOCK_M)
    col_offs = col_start + tl.arange(0, BLOCK_N)

    # Load 'a' (left half) and 'b' (right half) from x
    a_ptrs = x_ptr + (row_offs[:, None] * stride_x_m + col_offs[None, :] * stride_x_n)
    b_ptrs = x_ptr + (row_offs[:, None] * stride_x_m + (col_offs[None, :] + OFFSET_B) * stride_x_n)
    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)

    # Gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    x3 = a * a * a
    inner = GELU_COEFF * (a + GELU_CONST * x3)
    tanh_inner = libdevice.tanh(inner)
    gelu_a = 0.5 * a * (1.0 + tanh_inner)

    out = gelu_a * b

    out_ptrs = out_ptr + (row_offs[:, None] * stride_out_m + col_offs[None, :] * stride_out_n)
    tl.store(out_ptrs, out)

def triton_run(x: torch.Tensor) -> torch.Tensor:
    M, K = x.shape
    N = K // 2  # 4096
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # Tile dimensions that divide the output shape evenly
    BLOCK_M = 128
    BLOCK_N = 128

    grid = (M // BLOCK_M, N // BLOCK_N)
    geglu_kernel[grid](
        x, out,
        M, N,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        OFFSET_B=N,
        num_warps=4,
        num_stages=3,
    )
    return out