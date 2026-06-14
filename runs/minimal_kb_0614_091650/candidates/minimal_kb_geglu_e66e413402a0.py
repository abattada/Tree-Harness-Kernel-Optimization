import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ---------------------------------------------------------------------------
# Triton kernel: GeGLU (gated GELU, tanh approximation)
# Input:  x [M, K] with K even
# Output: out [M, K//2]
# One program per row, covers the whole output row with one block.
# No masks needed because BLOCK_SIZE == N.
# ---------------------------------------------------------------------------

@triton.jit
def geglu_kernel(
    x_ptr,                # pointer to input (2D flattened)
    out_ptr,              # pointer to output (2D flattened)
    stride_x_row,         # row stride of x (in elements)
    stride_out_row,       # row stride of out (in elements)
    N,                    # output dimension = K//2 (must equal BLOCK_SIZE)
    BLOCK_SIZE: tl.constexpr,
):
    # Static assertion: mask elimination only works when BLOCK_SIZE == N
    tl.static_assert(BLOCK_SIZE == N, "block size must equal output dimension")

    # ---- 1. row index ----------------------------------------------------
    row = tl.program_id(0).to(tl.int64)

    # ---- 2. pointers for this row ----------------------------------------
    x_row = x_ptr + row * stride_x_row
    out_row = out_ptr + row * stride_out_row

    # ---- 3. column indices (no mask needed) ------------------------------
    col_offs = tl.arange(0, BLOCK_SIZE)

    # ---- 4. load a and b (contiguous, no mask) --------------------------
    # a = first half of row, b = second half (starting at N)
    a = tl.load(x_row + col_offs)
    b = tl.load(x_row + col_offs + N)

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
    BLOCK_SIZE = N                      # covers the entire output row (N=4096)
    grid = (M,)                         # one program per row

    geglu_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,                    # good balance for full-row blocks
        num_stages=4,                   # enough to hide latency
    )
    return out