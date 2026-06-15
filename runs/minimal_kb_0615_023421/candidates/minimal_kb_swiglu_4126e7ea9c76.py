import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel: SwiGLU (SiLU gating)
# Input:  x [M, K]  with K even, K = 8192
# Output: out [M, K//2] = [M, 4096]
#
# This version takes advantage of the fact that BLOCK_SIZE equals N (4096),
# so all loads/stores are full rows and masks are dropped.
# Constexpr specialization, vectorized access, and eviction hints are used
# to maximize bandwidth.
# ---------------------------------------------------------------------------

@triton.jit
def swiglu_kernel(
    x_ptr,                # pointer to input (2D flattened)
    out_ptr,              # pointer to output (2D flattened)
    stride_x_row,         # row stride of x (in elements) = K
    stride_out_row,       # row stride of out (in elements) = N
    BLOCK_SIZE: tl.constexpr,   # ≡ N
):
    # ---- 1. row index ----------------------------------------------------
    row = tl.program_id(0)

    # ---- 2. pointers for this row ----------------------------------------
    x_row = x_ptr + row * stride_x_row
    out_row = out_ptr + row * stride_out_row

    # ---- 3. column indices (no mask needed: BLOCK_SIZE == N) ------------
    col_offs = tl.arange(0, BLOCK_SIZE)

    # ---- 4. load a and b, evict_first because both are read once ---------
    a = tl.load(x_row + col_offs, eviction_policy='evict_first')
    b = tl.load(x_row + col_offs + BLOCK_SIZE, eviction_policy='evict_first')

    # ---- 5. compute SiLU(a) = a * sigmoid(a) ------------------------------
    silu_a = a * tl.sigmoid(a)

    # ---- 6. gating --------------------------------------------------------
    out = silu_a * b

    # ---- 7. store result, evict_first (won't be read again) ---------------
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

    # Launch configuration
    BLOCK_SIZE = N                      # 4096, exactly covers one row
    grid = (M,)                         # one program per row

    swiglu_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,                    # good for register/occupancy balance
        num_stages=4,                   # moderate pipelining
    )

    return out