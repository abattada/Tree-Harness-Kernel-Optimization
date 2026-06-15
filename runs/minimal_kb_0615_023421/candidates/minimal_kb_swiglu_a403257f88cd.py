import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel: SwiGLU (silu gating) with grid‑stride loop
# Each program processes multiple rows (ROWS_PER_PROG) to amortize
# launch overhead and improve occupancy.
#
# Input:  x [M, K]  with K even, e.g. K = 8192
# Output: out [M, K//2] = silu(a) * b  where a,b = x.chunk(2, -1)
# ---------------------------------------------------------------------------

@triton.jit
def swiglu_kernel(
    x_ptr,                # pointer to input (2D, row-major)
    out_ptr,              # pointer to output (2D, row-major)
    stride_x_row,         # row stride of x (elements) = K
    stride_out_row,       # row stride of out (elements) = N
    BLOCK_SIZE: tl.constexpr,   # = N = K//2
    ROWS_PER_PROG: tl.constexpr, # number of rows handled per program
):
    # Each program processes a contiguous chunk of rows
    prog_id = tl.program_id(0)
    start_row = prog_id * ROWS_PER_PROG
    # Loop over rows assigned to this program
    for i in range(ROWS_PER_PROG):
        row = start_row + i
        # Pointer to this row
        x_row = x_ptr + row * stride_x_row
        out_row = out_ptr + row * stride_out_row

        col_offs = tl.arange(0, BLOCK_SIZE)

        # Load a and b (both halves of the row) – evict_first because read once
        a = tl.load(x_row + col_offs, eviction_policy='evict_first')
        b = tl.load(x_row + col_offs + BLOCK_SIZE, eviction_policy='evict_first')

        # silu(x) = x * sigmoid(x)
        sig_a = tl.sigmoid(a)
        silu_a = a * sig_a
        out = silu_a * b

        # Store result – evict_first (won't be reused)
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
    N = K // 2                           # output feature dimension

    out = torch.empty(M, N, device=x.device, dtype=torch.float32)

    # Each program processes 8 rows – M divisible by 8 (8192 / 8 = 1024 progs)
    ROWS_PER_PROG = 8
    grid = (M // ROWS_PER_PROG,)

    swiglu_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        BLOCK_SIZE=N,
        ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=8,
        num_stages=4,
    )
    return out