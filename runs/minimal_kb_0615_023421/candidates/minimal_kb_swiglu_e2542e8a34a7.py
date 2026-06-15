import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel: SwiGLU (silu gating)
# Input:  x [M, K]  with K even, e.g. K = 8192
# Output: out [M, K//2] = silu(a) * b  where a,b = x.chunk(2, -1)
#
# Optimizations:
#   - Full row blocks (BLOCK_SIZE = N = K//2) so no masking needed.
#   - Eviction policy "evict_first" for streaming accesses.
#   - constexpr hints (tl.static_assert, tl.multiple_of) to enable
#     better vectorized code generation.
#   - Moderate num_warps=8 to balance register use and occupancy.
# ---------------------------------------------------------------------------

@triton.jit
def swiglu_kernel(
    x_ptr,                # pointer to input (2D, row-major)
    out_ptr,              # pointer to output (2D, row-major)
    stride_x_row,         # row stride of x (in elements) = K
    stride_out_row,       # row stride of out (in elements) = N
    BLOCK_SIZE: tl.constexpr,   # = N = K//2
):
    # ---- 1. row index ----------------------------------------------------
    row = tl.program_id(0)

    # ---- 2. pointers for this row ----------------------------------------
    # Use tl.multiple_of to inform the compiler that pointers are aligned.
    x_row = tl.multiple_of(x_ptr + row * stride_x_row, 16)
    out_row = tl.multiple_of(out_ptr + row * stride_out_row, 16)

    # ---- 3. column indices (BLOCK_SIZE is exactly N, no mask needed) ----
    col_offs = tl.arange(0, BLOCK_SIZE)

    # ---- 4. load a and b, evict_first because both are read once ---------
    a = tl.load(x_row + col_offs, eviction_policy='evict_first')
    b = tl.load(x_row + col_offs + BLOCK_SIZE, eviction_policy='evict_first')

    # ---- 5. compute silu(a) = a * sigmoid(a) -----------------------------
    sig_a = tl.sigmoid(a)
    silu_a = a * sig_a

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

    # Launch configuration: one program per row, full-row block
    BLOCK_SIZE = N                      # 4096 for the given shape
    grid = (M,)                         # one program per row

    swiglu_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,                    # decreased from 16 to reduce register pressure and increase occupancy
        num_stages=4,                   # moderate pipeline depth
    )
    return out