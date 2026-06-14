import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ---------------------------------------------------------------------------
# Triton kernel: GeGLU (gated GELU, tanh approximation)
# Input:  x [M, K]   (K even)
# Output: out [M, K//2]
# Each program processes one row, one contiguous block of N elements.
# ---------------------------------------------------------------------------

@triton.jit
def geglu_kernel(
    x_ptr,          # input tensor pointer, 2D flattened
    out_ptr,        # output tensor pointer, 2D flattened
    stride_x,       # row stride of x (in elements) = K
    stride_out,     # row stride of out (in elements) = N
    N,              # output dimension = K//2
    BLOCK_SIZE: tl.constexpr,   # number of columns per program (must equal N)
):
    # Ensure exact coverage (no mask needed when shapes divide evenly)
    tl.static_assert(N % BLOCK_SIZE == 0)

    row = tl.program_id(0)                    # row index
    x_row = x_ptr + row * stride_x
    out_row = out_ptr + row * stride_out

    col_offs = tl.arange(0, BLOCK_SIZE)        # 0 .. BLOCK_SIZE-1

    # Load a (first half) and b (second half)
    a = tl.load(x_row + col_offs)              # no mask required
    b = tl.load(x_row + col_offs + N)          # b starts at column N

    # Compute GELU with tanh approximation
    sqrt_2_over_pi = 0.7978845608028654        # sqrt(2/π)
    c = 0.044715
    x_cube = a * a * a
    inner = sqrt_2_over_pi * (a + c * x_cube)
    tanh_val = libdevice.tanh(inner)
    gelu = 0.5 * a * (1.0 + tanh_val)

    result = gelu * b

    # Store result
    tl.store(out_row + col_offs, result)


# ---------------------------------------------------------------------------
# Public entry point – allocates output and launches the kernel.
# ---------------------------------------------------------------------------

def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    x: float32 tensor of shape [M, K]  (K even)
    returns: float32 tensor of shape [M, K//2]
    """
    M, K = x.shape
    N = K // 2                              # output feature dimension

    out = torch.empty(M, N, device=x.device, dtype=torch.float32)

    # Kernel configuration: one program per row, process entire row in one block
    BLOCK_SIZE = N                          # 4096, covers the entire output row
    grid = (M,)                            # one program per row

    geglu_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=4,
    )
    return out