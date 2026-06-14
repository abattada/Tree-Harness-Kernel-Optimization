import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ---------------------------------------------------------------------------
# Triton kernel: GeGLU (gated GELU with tanh approximation)
# Input:  x [M, K] with K even
# Output: out [M, K//2]
# One program per row, covering the whole output row in one block.
# ---------------------------------------------------------------------------

@triton.jit
def geglu_kernel(
    x_ptr,                # pointer to input (2D flattened)
    out_ptr,              # pointer to output (2D flattened)
    stride_x_row,         # row stride of x (in elements)
    stride_out_row,       # row stride of out (in elements)
    N,                    # output dimension = K//2
    BLOCK_SIZE: tl.constexpr,   # number of elements per block (must equal N for full row)
):
    # ---- 1. row index ----------------------------------------------------
    row = tl.program_id(0)

    # ---- 2. pointers for this row ----------------------------------------
    x_row = x_ptr + row * stride_x_row
    out_row = out_ptr + row * stride_out_row

    # ---- 3. column indices and mask --------------------------------------
    col_offs = tl.arange(0, BLOCK_SIZE)
    mask = col_offs < N          # safety mask, but if BLOCK_SIZE == N it's always true

    # ---- 4. load a and b -------------------------------------------------
    # a = first half of row, b = second half (starting at N)
    a = tl.load(x_row + col_offs, mask=mask, other=0.0)
    b = tl.load(x_row + col_offs + N, mask=mask, other=0.0)

    # ---- 5. compute GELU_tanh(a) -----------------------------------------
    # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt2_over_pi = 0.7978845608028654
    c = 0.044715

    a3 = a * a * a
    inner = sqrt2_over_pi * (a + c * a3)
    tanh_val = libdevice.tanh(inner)
    gelu_a = 0.5 * a * (1.0 + tanh_val)

    # ---- 6. gating --------------------------------------------------------
    out = gelu_a * b

    # ---- 7. store result -------------------------------------------------
    tl.store(out_row + col_offs, out, mask=mask)


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

    # Launch configuration
    BLOCK_SIZE = 4096                   # covers the entire output row (N=4096)
    grid = (M,)                         # one program per row

    geglu_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,                    # tuning knob: try 4, 8, 16
        num_stages=4,                   # tuning knob: try 2..6
    )
    return out