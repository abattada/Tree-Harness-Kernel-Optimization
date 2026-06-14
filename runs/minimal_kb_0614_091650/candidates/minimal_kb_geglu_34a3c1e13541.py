import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ---------------------------------------------------------------------------
# Triton kernel: GeGLU (gated GELU with tanh approximation)
# Input:  x  [M, K]   (K even)
# Output: out [M, K//2]
# 2D grid: one program per (row, output column block)
# BLOCK_SIZE divides N = K//2 exactly.
# ---------------------------------------------------------------------------

@triton.jit
def geglu_kernel(
    x_ptr,          # pointer to input (2D flattened)
    out_ptr,        # pointer to output (2D flattened)
    stride_x,       # row stride of x (in elements)   = K
    stride_out,     # row stride of out (in elements) = K//2
    N,              # output dimension = K//2
    BLOCK_SIZE: tl.constexpr,   # number of output elements per block
):
    # ---- 1. program indices ------------------------------------------------
    row = tl.program_id(0).to(tl.int64)
    col_block = tl.program_id(1)

    # ---- 2. column offsets and mask ----------------------------------------
    col_offs = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # N is the output dimension, always divides BLOCK_SIZE exactly by construction
    mask = col_offs < N   # always true for perfect multiple; kept for safety

    # ---- 3. pointers for this row (base addresses) -------------------------
    x_base = x_ptr + row * stride_x
    out_base = out_ptr + row * stride_out

    # ---- 4. load a (first half) and b (second half) ------------------------
    a = tl.load(x_base + col_offs, mask=mask, other=0.0)
    b = tl.load(x_base + col_offs + N, mask=mask, other=0.0)   # b starts at N

    # ---- 5. compute GELU with tanh approximation ---------------------------
    #   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654   # sqrt(2/pi)
    c = 0.044715

    x_cube = a * a * a
    inner = sqrt_2_over_pi * (a + c * x_cube)
    tanh_val = libdevice.tanh(inner)
    gelu_a = 0.5 * a * (1.0 + tanh_val)

    # ---- 6. gating ---------------------------------------------------------
    result = gelu_a * b

    # ---- 7. store result ---------------------------------------------------
    tl.store(out_base + col_offs, result, mask=mask)


# ---------------------------------------------------------------------------
# Public entry point – allocates output and launches the kernel.
# ---------------------------------------------------------------------------

def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    x: float32 tensor of shape [M, K]  (K even)
    returns: float32 tensor of shape [M, K//2]
    """
    M, K = x.shape
    N = K // 2                          # output feature dimension

    # Allocate output
    out = torch.empty(M, N, device=x.device, dtype=torch.float32)

    # Launch configuration – tuned for occupancy
    BLOCK_SIZE = 1024                   # output elements per block; divides N exactly
    grid = (M, N // BLOCK_SIZE)         # one program per row and column block

    geglu_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=4,
    )
    return out