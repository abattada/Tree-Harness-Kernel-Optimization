import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ---------------------------------------------------------------------------
# GeGLU kernel (gated GELU with tanh approximation)
# Input:  x [M, K] with K even (here [8192, 8192])
# Output: out [M, K//2] = [8192, 4096]
# One program per row; BLOCK_SIZE = N = K//2, so the full output row is
# processed without masks.  The design leaves the obvious tuning knobs:
# num_warps, num_stages, and (if shapes become dynamic) block size.
# ---------------------------------------------------------------------------

@triton.jit
def geglu_kernel(
    x_ptr,                # pointer to input (2D, flat)
    out_ptr,              # pointer to output (2D, flat)
    stride_x,             # row stride of x in elements (= K)
    stride_out,           # row stride of out in elements (= N)
    BLOCK_SIZE: tl.constexpr,   # must equal N for the full-row path
    N: tl.constexpr,            # output dimension = K//2
):
    # Ensure we have exactly one block per row (no partial tiles)
    tl.static_assert(BLOCK_SIZE == N,
                     "BLOCK_SIZE must equal N for this specialization")

    # Row index from launch grid
    row = tl.program_id(0)

    # Column offsets for the entire output row (no mask needed)
    col_offs = tl.arange(0, BLOCK_SIZE)

    # Row pointers
    x_row = x_ptr + row * stride_x
    out_row = out_ptr + row * stride_out

    # Load first half (a) and second half (b) – evict_first because they are
    # read exactly once in this kernel invocation.
    a = tl.load(x_row + col_offs,           eviction_policy='evict_first')
    b = tl.load(x_row + col_offs + N,       eviction_policy='evict_first')

    # GELU(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
    sqrt2_over_pi = 0.7978845608028654
    c = 0.044715

    a3 = a * a * a
    inner = sqrt2_over_pi * (a + c * a3)
    tanh_val = libdevice.tanh(inner)
    gelu_a = 0.5 * a * (1.0 + tanh_val)

    # Gating
    out = gelu_a * b

    # Store result – evict_first since the output will not be read again here
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
    N = K // 2                  # 4096 for the given input

    # Allocate output
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)

    # Launch configuration: one program per row, block size == N
    BLOCK_SIZE = N
    grid = (M,)

    geglu_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        N=N,
        num_warps=8,            # good balance for this register usage
        num_stages=4,           # moderate pipelining
    )

    return out