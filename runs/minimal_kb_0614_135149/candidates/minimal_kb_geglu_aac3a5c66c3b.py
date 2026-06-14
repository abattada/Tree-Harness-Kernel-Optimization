import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ---------------------------------------------------------------------------
# GeGLU kernel – gated GELU with tanh approximation
# Input:  x [M, K] with K even (here [8192, 8192])
# Output: out [M, K//2] = [8192, 4096]
# One program per row, full row in a block (BLOCK_SIZE = N).  No masks.
# The main refinement over the parent: increased num_warps to 16 for
# better latency hiding and lowered num_stages to 3 to keep register
# pressure in check.
# ---------------------------------------------------------------------------

@triton.jit
def geglu_kernel(
    x_ptr,                # pointer to input (2D flattened)
    out_ptr,              # pointer to output (2D flattened)
    stride_x,             # row stride of x (in elements)
    stride_out,           # row stride of out (in elements)
    BLOCK_SIZE: tl.constexpr,   # = N (output dim), must be 4096 for given shape
):
    # One program per row – BLOCK_SIZE exactly covers the output row,
    # so no boundary masking is needed.
    row = tl.program_id(0)
    col_offs = tl.arange(0, BLOCK_SIZE)

    # Row pointers
    x_row = x_ptr + row * stride_x
    out_row = out_ptr + row * stride_out

    # Load first half (a) and second half (b)
    # evict_first since they are used exactly once here.
    a = tl.load(x_row + col_offs, eviction_policy='evict_first')
    b = tl.load(x_row + col_offs + BLOCK_SIZE, eviction_policy='evict_first')

    # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt2_over_pi = 0.7978845608028654
    c = 0.044715

    a3 = a * a * a
    inner = sqrt2_over_pi * (a + c * a3)
    tanh_val = libdevice.tanh(inner)
    gelu_a = 0.5 * a * (1.0 + tanh_val)

    # Gating
    out = gelu_a * b

    # Store result – evict_first because output is written only once
    tl.store(out_row + col_offs, out, eviction_policy='evict_first')


# ---------------------------------------------------------------------------
# Public entry point – allocates output and launches the kernel.
# ---------------------------------------------------------------------------

def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    x: float32 tensor of shape [8192, 8192]
    returns: float32 tensor of shape [8192, 4096]
    """
    M, K = x.shape
    N = K // 2                      # 4096

    # Allocate output
    out = torch.empty(M, N, device=x.device, dtype=torch.float32)

    # Launch configuration: one program per row, block size = N
    geglu_kernel[(M,)](
        x, out,
        x.stride(0), out.stride(0),
        BLOCK_SIZE=N,               # constexpr 4096
        num_warps=16,               # increased from 8 to 16
        num_stages=3,               # decreased from 4 to 3
    )

    return out