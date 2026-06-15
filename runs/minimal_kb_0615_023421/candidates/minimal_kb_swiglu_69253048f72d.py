import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel: SwiGLU
# Input:  x [M, K] with K even (K = 8192)
# Output: out [M, K//2] = [M, 4096]
#
# This version uses a 2D grid: one program per (row, column tile).
# BLOCK_SIZE is chosen to divide N=4096 evenly, so no column mask needed.
# Eviction hints are set to 'evict_first' since all data is used once.
# ---------------------------------------------------------------------------

@triton.jit
def swiglu_kernel(
    x_ptr,               # pointer to input (2D, row-major)
    out_ptr,             # pointer to output (2D, row-major)
    stride_x_row,        # stride between rows in input (in elements)
    stride_out_row,      # stride between rows in output (in elements)
    N: tl.constexpr,     # output dimension = K//2
    BLOCK_SIZE: tl.constexpr,
):
    # ---- 1. compute row and column tile indices ------------------------
    row = tl.program_id(0)
    col_tile = tl.program_id(1)
    col_start = col_tile * BLOCK_SIZE

    # ---- 2. pointers for this tile ------------------------------------
    x_base = x_ptr + row * stride_x_row
    out_base = out_ptr + row * stride_out_row

    # ---- 3. column offsets (no mask: BLOCK_SIZE divides N) ------------
    col_offs = tl.arange(0, BLOCK_SIZE)

    # ---- 4. load a and b from the two halves of the row ---------------
    # a lives at columns [0, N), b at columns [N, 2N)
    a = tl.load(x_base + col_start + col_offs,
                eviction_policy='evict_first')
    b = tl.load(x_base + N + col_start + col_offs,
                eviction_policy='evict_first')

    # ---- 5. compute silu(a) = a * sigmoid(a) --------------------------
    sig_a = tl.sigmoid(a)
    silu_a = a * sig_a

    # ---- 6. gating ----------------------------------------------------
    out = silu_a * b

    # ---- 7. store result ----------------------------------------------
    tl.store(out_base + col_start + col_offs, out,
             eviction_policy='evict_first')


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
    assert K % 2 == 0, "K must be even"

    # Allocate output
    out = torch.empty(M, N, device=x.device, dtype=torch.float32)

    # Launch configuration
    BLOCK_SIZE = 1024                   # divides N=4096 exactly
    grid = (M, N // BLOCK_SIZE)         # one program per row × column tile

    swiglu_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,                    # smaller blocks need fewer warps
        num_stages=4,                   # moderate pipelining
    )

    return out