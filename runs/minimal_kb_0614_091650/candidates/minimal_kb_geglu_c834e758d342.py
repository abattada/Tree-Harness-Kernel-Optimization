import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ---------------------------------------------------------------------------
# Triton kernel: GeGLU (gated GELU with tanh approximation)
# Input:  x [M, K] with K even, K=8192
# Output: out [M, K//2] = [M, 4096]
# Multiple rows per program (ROWS_PER_PROG=4) to reduce launch overhead
# ---------------------------------------------------------------------------

@triton.jit
def geglu_kernel(
    x_ptr,                # pointer to input (2D flattened)
    out_ptr,              # pointer to output (2D flattened)
    stride_x,             # row stride of x (in elements) = K
    stride_out,           # row stride of out (in elements) = K//2
    M,                    # number of rows
    BLOCK_SIZE: tl.constexpr,   # = 4096 (output dimension)
    ROWS_PER_PROG: tl.constexpr, # = 4
):
    prog_id = tl.program_id(0)
    start_row = prog_id * ROWS_PER_PROG

    # Loop over rows assigned to this program
    for i in range(ROWS_PER_PROG):
        row = start_row + i
        if row < M:
            # Row pointers
            x_row = x_ptr + row * stride_x
            out_row = out_ptr + row * stride_out

            # Column offsets (0..BLOCK_SIZE-1)
            col_offs = tl.arange(0, BLOCK_SIZE)

            # Load a (first half) and b (second half) – no mask needed since
            # BLOCK_SIZE equals N exactly, and rows are full-length.
            a = tl.load(x_row + col_offs, eviction_policy='evict_first')
            b = tl.load(x_row + col_offs + BLOCK_SIZE, eviction_policy='evict_first')

            # Compute GELU_tanh(a)
            sqrt2_over_pi = 0.7978845608028654
            c = 0.044715
            a3 = a * a * a
            inner = sqrt2_over_pi * (a + c * a3)
            tanh_val = libdevice.tanh(inner)
            gelu_a = 0.5 * a * (1.0 + tanh_val)

            # Gating: out = gelu(a) * b
            out = gelu_a * b

            # Store result
            tl.store(out_row + col_offs, out, eviction_policy='evict_first')


# ---------------------------------------------------------------------------
# Public entry point – allocates output and launches the kernel.
# ---------------------------------------------------------------------------

def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    x: float32 tensor of shape [M, K] with K even (K=8192)
    returns: float32 tensor of shape [M, K//2]
    """
    M, K = x.shape
    N = K // 2                           # 4096

    # Allocate output
    out = torch.empty(M, N, device=x.device, dtype=torch.float32)

    # Launch configuration
    BLOCK_SIZE = N                       # process entire output row in one block
    ROWS_PER_PROG = 4
    grid = ((M + ROWS_PER_PROG - 1) // ROWS_PER_PROG,)   # ceil(M / 4)

    geglu_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        M,
        BLOCK_SIZE=BLOCK_SIZE,
        ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=8,
        num_stages=4,
    )
    return out