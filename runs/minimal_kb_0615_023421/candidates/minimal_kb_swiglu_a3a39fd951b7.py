import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel: SWiGLU (silu(a) * b)
# Input:  x [M, K] with K even, K = 8192, M = 8192
# Output: out [M, K//2] = [M, 4096]
#
# Improved from seed:
#   - Made M, K, N constexpr to let the compiler fully specialize
#     and eliminate any shape-related checks.
#   - Used tl.max_contiguous on the column offsets to hint maximal
#     vectorized memory accesses.
#   - All loads/stores are full rows with no masking needed (BLOCK_SIZE == N).
# ---------------------------------------------------------------------------

@triton.jit
def swiglu_kernel(
    x_ptr,                # pointer to input (2D flattened)
    out_ptr,              # pointer to output (2D flattened)
    stride_x_row,         # row stride of x (in elements) = K
    stride_out_row,       # row stride of out (in elements) = N
    M: tl.constexpr,      # number of rows (8192)
    K: tl.constexpr,      # input dimension (8192)
    N: tl.constexpr,      # output dimension (4096)
    BLOCK_SIZE: tl.constexpr,   # ≡ N = 4096
):
    # ---- 1. row index ----------------------------------------------------
    row = tl.program_id(0)

    # ---- 2. column offsets (contiguous, aligned, no mask needed) -------
    col_offs = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)

    # ---- 3. base pointers for this row ----------------------------------
    x_row = x_ptr + row * stride_x_row
    out_row = out_ptr + row * stride_out_row

    # ---- 4. load a and b (both contiguous, each read once) --------------
    a = tl.load(x_row + col_offs, eviction_policy='evict_first')
    b = tl.load(x_row + col_offs + BLOCK_SIZE, eviction_policy='evict_first')

    # ---- 5. compute SiLU(a) = a * sigmoid(a) ----------------------------
    silu_a = a * tl.sigmoid(a)      # no explicit tl.silu in Triton 3.x

    # ---- 6. gating: silu(a) * b -----------------------------------------
    out = silu_a * b

    # ---- 7. store result, evict_first (won't be read again) -------------
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
        M, K, N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,                    # good balance occupancy vs. registers
        num_stages=4,                   # moderate pipelining
    )
    return out