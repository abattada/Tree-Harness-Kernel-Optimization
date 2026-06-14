import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ---------------------------------------------------------------------------
# Triton kernel: GeGLU (gated GELU with tanh approximation)
# Input:  x [M, K] with K even (M=8192, K=8192)
# Output: out [M, K//2] = [8192, 4096]
#
# This version processes multiple rows per program (ROWS_PER_PROG=4) to reduce
# the grid size and thus launch overhead.  Each program loops over its assigned
# rows.  Full-row tiles, no masks needed.
# ---------------------------------------------------------------------------

@triton.jit
def geglu_kernel(
    x_ptr,                # pointer to input (2D flattened)
    out_ptr,              # pointer to output (2D flattened)
    stride_x_row,         # row stride of x (in elements)
    stride_out_row,       # row stride of out (in elements)
    ROWS_PER_PROG: tl.constexpr,   # number of consecutive rows per program
    BLOCK_SIZE: tl.constexpr,      # ≡ N (output dimension)
    N: tl.constexpr,               # output dimension (K//2)
):
    # Static assertions
    tl.static_assert(BLOCK_SIZE == N,
                     "BLOCK_SIZE must equal N for this specialization")
    # Row offset for this program
    base_row = tl.program_id(0) * ROWS_PER_PROG

    # Loop over the assigned rows
    for i in tl.static_range(ROWS_PER_PROG):
        row = base_row + i

        # Pointers to this row
        x_row = x_ptr + row * stride_x_row
        out_row = out_ptr + row * stride_out_row

        # Column indices (always valid)
        col_offs = tl.arange(0, BLOCK_SIZE)

        # Load a (first half) and b (second half) – evict_first
        a = tl.load(x_row + col_offs, eviction_policy='evict_first')
        b = tl.load(x_row + col_offs + N, eviction_policy='evict_first')

        # Compute GELU_tanh(a)
        sqrt2_over_pi = 0.7978845608028654
        c = 0.044715

        a3 = a * a * a
        inner = sqrt2_over_pi * (a + c * a3)
        tanh_val = libdevice.tanh(inner)
        gelu_a = 0.5 * a * (1.0 + tanh_val)

        # Gate
        out = gelu_a * b

        # Store result (evict_first)
        tl.store(out_row + col_offs, out, eviction_policy='evict_first')


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

    # Tuning constants
    ROWS_PER_PROG = 4                   # must divide M
    assert M % ROWS_PER_PROG == 0

    BLOCK_SIZE = N                      # 4096, exactly one output row
    grid = (M // ROWS_PER_PROG,)        # fewer programs

    geglu_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        ROWS_PER_PROG=ROWS_PER_PROG,
        BLOCK_SIZE=BLOCK_SIZE,
        N=N,
        num_warps=8,
        num_stages=4,                   # kept for consistency (no effect)
    )

    return out