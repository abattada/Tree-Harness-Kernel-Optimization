import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ---------------------------------------------------------------------------
# Persistent GeGLU kernel: one program per SM, each processing multiple rows.
# This reduces launch overhead and improves occupancy on large-batch inputs.
# ---------------------------------------------------------------------------

@triton.jit
def geglu_kernel(
    x_ptr,
    out_ptr,
    stride_x,
    stride_out,
    BLOCK_SIZE: tl.constexpr,       # = N  (output dimension)
    N: tl.constexpr,                 # output dimension, equals K//2
    ROWS_PER_PROGRAM: tl.constexpr,  # number of rows per program block
):
    # Static assertions for full‑tile specialization
    tl.static_assert(BLOCK_SIZE == N, "BLOCK_SIZE must equal N for full rows")

    # Starting row for this program
    start_row = tl.program_id(0) * ROWS_PER_PROGRAM

    # Column indices (valid for all rows)
    col_offs = tl.arange(0, BLOCK_SIZE)

    # Loop over the assigned chunk of rows
    for i in range(ROWS_PER_PROGRAM):
        row = start_row + i

        # Pointers for this row
        x_row = x_ptr + row * stride_x
        out_row = out_ptr + row * stride_out

        # Load a (first half) and b (second half) – evict_first because read once
        a = tl.load(x_row + col_offs, eviction_policy='evict_first')
        b = tl.load(x_row + col_offs + N, eviction_policy='evict_first')

        # GELU_tanh(a) formula
        sqrt2_over_pi = 0.7978845608028654
        c = 0.044715
        a3 = a * a * a
        inner = sqrt2_over_pi * (a + c * a3)
        tanh_val = libdevice.tanh(inner)
        gelu_a = 0.5 * a * (1.0 + tanh_val)

        # Gating
        out = gelu_a * b

        # Store result (write once, evict before next row)
        tl.store(out_row + col_offs, out, eviction_policy='evict_first')


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    x: float32 tensor of shape [M, K]  (K even)
    returns: float32 tensor of shape [M, K//2]
    """
    M, K = x.shape
    N = K // 2

    # Choose grid size and rows per program to cover all rows exactly.
    # Here M=8192, we use 128 programs (common SM count) each handling 64 rows.
    NUM_PROGRAMS = 128
    ROWS_PER_PROGRAM = M // NUM_PROGRAMS   # 64
    assert M % NUM_PROGRAMS == 0, "M must be divisible by NUM_PROGRAMS"

    out = torch.empty(M, N, device=x.device, dtype=torch.float32)

    grid = (NUM_PROGRAMS,)
    geglu_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        BLOCK_SIZE=N,
        N=N,
        ROWS_PER_PROGRAM=ROWS_PER_PROGRAM,
        num_warps=8,
        num_stages=4,
    )

    return out