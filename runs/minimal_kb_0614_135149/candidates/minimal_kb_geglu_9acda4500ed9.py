import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ---------------------------------------------------------------------------
# Triton kernel: GeGLU (gated GELU with tanh approximation)
# Input:  x [M, K] with K = 8192, M = 8192
# Output: out [M, K//2] = [8192, 4096]
# Multi-row refinement: each program processes ROWS_PER_PROGRAM=8 rows.
# This reduces launch overhead and improves occupancy when combined with
# moderate warp/stage counts.
# ---------------------------------------------------------------------------

@triton.jit
def geglu_kernel(
    x_ptr,                      # pointer to input [M, K] (flattened)
    out_ptr,                    # pointer to output [M, N] (flattened)
    stride_x_row,               # row stride of x in elements = K
    stride_out_row,             # row stride of out in elements = N
    BLOCK_SIZE: tl.constexpr,   # must equal N (output width)
    N: tl.constexpr,            # output width = K // 2
    ROWS_PER_PROGRAM: tl.constexpr,
    M: tl.constexpr,            # number of rows (needed for static assert)
):
    # Safety: our block size exactly covers one output row → no masks
    tl.static_assert(BLOCK_SIZE == N,
                     "BLOCK_SIZE must equal N for this specialization")
    # We only launch programs that cover full rows; no partial tail if M is
    # divisible by ROWS_PER_PROGRAM (true for 8192 / 8)
    tl.static_assert(M % ROWS_PER_PROGRAM == 0,
                     "M must be a multiple of ROWS_PER_PROGRAM")

    # Program id and starting row
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROGRAM

    # Column offsets (always valid: full row)
    col_offs = tl.arange(0, BLOCK_SIZE)

    # Process a batch of rows in this program
    for r in range(ROWS_PER_PROGRAM):
        row = row_start + r

        # Pointers for this row
        x_row = x_ptr + row * stride_x_row
        out_row = out_ptr + row * stride_out_row

        # Load left half (a) and right half (b) – streaming, evict after use
        a = tl.load(x_row + col_offs, eviction_policy='evict_first')
        b = tl.load(x_row + col_offs + N, eviction_policy='evict_first')

        # GELU with tanh approximation:
        #   GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        sqrt2_over_pi = 0.7978845608028654
        c = 0.044715

        a3 = a * a * a
        inner = sqrt2_over_pi * (a + c * a3)
        tanh_val = libdevice.tanh(inner)
        gelu_a = 0.5 * a * (1.0 + tanh_val)

        # Gate with b
        out = gelu_a * b

        # Store result (streaming write)
        tl.store(out_row + col_offs, out, eviction_policy='evict_first')


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    x: float32 tensor of shape [8192, 8192]
    returns: float32 tensor of shape [8192, 4096]
    """
    M, K = x.shape
    N = K // 2                     # 4096
    BLOCK_SIZE = N                 # one block covers a full output row

    # Each program handles 8 rows to reduce launch overhead.
    # 8192 / 8 = 1024 programs, perfect occupancy with 8 warps/block.
    ROWS_PER_PROGRAM = 8
    grid = (M // ROWS_PER_PROGRAM,)

    out = torch.empty(M, N, device=x.device, dtype=torch.float32)

    geglu_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        N=N,
        ROWS_PER_PROGRAM=ROWS_PER_PROGRAM,
        M=M,
        num_warps=8,
        num_stages=4,
    )

    return out