import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ---------------------------------------------------------------------------
# Triton kernel: GeGLU (gated GELU with tanh approximation)
# Input:  x [M, K] with K even, e.g., (8192, 8192)
# Output: out [M, K//2] = (8192, 4096)
#
# This version processes multiple rows per program (ROWS_PER_PROG) to
# reduce launch overhead and improve cache reuse. Each program handles
# a contiguous block of ROWS_PER_PROG rows, iterating over rows in a
# small loop. Full-row tiles (BLOCK_SIZE = N) eliminate boundary masks.
# ---------------------------------------------------------------------------

@triton.jit
def geglu_kernel(
    x_ptr,               # pointer to input (2D)
    out_ptr,             # pointer to output (2D)
    stride_x_row,        # row stride of x (in elements)
    stride_out_row,      # row stride of out (in elements)
    BLOCK_SIZE: tl.constexpr,   # = N  (output dimension)
    N: tl.constexpr,            # output dimension (same as BLOCK_SIZE)
    ROWS_PER_PROG: tl.constexpr, # number of rows handled by one program
):
    # Static assertion to guarantee no partial tiles
    tl.static_assert(BLOCK_SIZE == N, "BLOCK_SIZE must equal N for full-row tiles")

    # Base row index for this program
    base_row = tl.program_id(0) * ROWS_PER_PROG

    # Column indices (same for all rows; no mask needed)
    col_offs = tl.arange(0, BLOCK_SIZE)

    # Loop over rows in the assigned chunk
    for r in range(ROWS_PER_PROG):
        row = base_row + r

        # Pointers to this row in input and output
        x_row = x_ptr + row * stride_x_row
        out_row = out_ptr + row * stride_out_row

        # Load a (first half) and b (second half) – evict_first because they
        # are read only once, reducing L2 pressure.
        a = tl.load(x_row + col_offs, eviction_policy='evict_first')
        b = tl.load(x_row + col_offs + N, eviction_policy='evict_first')

        # Compute GELU_tanh(a)
        # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        sqrt2_over_pi = 0.7978845608028654
        c = 0.044715

        a3 = a * a * a
        inner = sqrt2_over_pi * (a + c * a3)
        tanh_val = libdevice.tanh(inner)
        gelu_a = 0.5 * a * (1.0 + tanh_val)

        # Gate
        out = gelu_a * b

        # Store result – evict_first since it is written once and not read back
        tl.store(out_row + col_offs, out, eviction_policy='evict_first')


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    x: float32 tensor of shape [M, K] (K even)
    Returns: float32 tensor of shape [M, K//2]
    """
    M, K = x.shape
    N = K // 2  # output dimension

    # Allocate output
    out = torch.empty(M, N, device=x.device, dtype=torch.float32)

    # Launch configuration
    BLOCK_SIZE = 4096         # equals N for the given shape
    ROWS_PER_PROG = 8         # process 8 rows per program to reduce launches
    grid = (M // ROWS_PER_PROG,)

    geglu_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        N=N,
        ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=8,
        num_stages=4,
    )

    return out