import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel: SwiGLU (silu gating) with 2D tile processing
# Input:  x [M, K]  with K even (e.g., 8192)
# Output: out [M, K//2] = silu(a) * b  where a,b = x.chunk(2, -1)
#
# This version processes BLOCK_M rows per program (2D tile) to reduce
# the number of grid programs and improve SM utilization.
# ---------------------------------------------------------------------------

@triton.jit
def swiglu_kernel(
    x_ptr,                # pointer to input (2D, row-major)
    out_ptr,              # pointer to output (2D, row-major)
    stride_x_row,         # row stride of x (in elements) = K
    stride_out_row,       # row stride of out (in elements) = N
    M,                    # total rows (runtime)
    N: tl.constexpr,      # output dimension (K//2)
    BLOCK_M: tl.constexpr,# number of rows per program
):
    # Program id and base row
    pid = tl.program_id(0)
    row_base = pid * BLOCK_M

    # Row and column offsets for the 2D tile
    row_offsets = tl.arange(0, BLOCK_M)
    col_offsets = tl.arange(0, N)

    # Pointer to the start of the tile in x and out
    x_tile = x_ptr + row_base * stride_x_row
    out_tile = out_ptr + row_base * stride_out_row

    # Load a (first half) and b (second half) for all BLOCK_M rows
    # a: columns 0..N-1, b: columns N..2N-1
    a = tl.load(
        x_tile + row_offsets[:, None] * stride_x_row + col_offsets[None, :],
        eviction_policy='evict_first'
    )
    b = tl.load(
        x_tile + row_offsets[:, None] * stride_x_row + (col_offsets + N)[None, :],
        eviction_policy='evict_first'
    )

    # silu(a) = a * sigmoid(a)
    silu_a = a * tl.sigmoid(a)

    # Gating
    out = silu_a * b

    # Store result
    tl.store(
        out_tile + row_offsets[:, None] * stride_out_row + col_offsets[None, :],
        out,
        eviction_policy='evict_first'
    )


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

    # Launch configuration: 2D tile parameters
    BLOCK_SIZE = N                      # 4096, exactly covers half a row
    BLOCK_M = 4                         # rows per program (tunable)
    grid = (M // BLOCK_M,)              # number of programs (assumes M divisible)

    swiglu_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        M,
        BLOCK_SIZE,
        BLOCK_M=BLOCK_M,
        num_warps=8,
        num_stages=4,
    )
    return out