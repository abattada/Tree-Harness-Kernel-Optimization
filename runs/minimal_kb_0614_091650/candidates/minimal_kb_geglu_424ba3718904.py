import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ---------------------------------------------------------------------------
# Triton kernel: GeGLU (gated GELU with tanh approximation)
# Input:  x [M, K] with K even
# Output: out [M, K//2]
# One program per row, covering the whole output row in one block.
# Specialized for the case BLOCK_SIZE == N (full row) to remove masks.
# ---------------------------------------------------------------------------

@triton.jit
def geglu_kernel(
    x_ptr,                # pointer to input (2D flattened)
    out_ptr,              # pointer to output (2D flattened)
    stride_x,             # row stride of x (in elements)
    stride_out,           # row stride of out (in elements)
    BLOCK_SIZE: tl.constexpr,   # must equal N
    N: tl.constexpr,            # output dimension = K//2
):
    # Assert that the block covers a full output row (no partial tile)
    tl.static_assert(BLOCK_SIZE == N,
                     "BLOCK_SIZE must equal N for this specialization")

    # ---- 1. row index ----------------------------------------------------
    row = tl.program_id(0)

    # ---- 2. column indices (always valid; no mask needed) ----------------
    col_offs = tl.arange(0, BLOCK_SIZE)

    # ---- 3. pointers for this row ----------------------------------------
    x_row = x_ptr + row * stride_x
    out_row = out_ptr + row * stride_out

    # ---- 4. load a (first half) and b (second half) ----------------------
    a = tl.load(x_row + col_offs)               # no mask needed
    b = tl.load(x_row + col_offs + N)           # no mask needed

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

    # ---- 7. store result (no mask) ---------------------------------------
    tl.store(out_row + col_offs, out)


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
        BLOCK_SIZE=BLOCK_SIZE,
        N=N,
        num_warps=8,
        num_stages=4,
    )
    return out