import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel: SWiGLU
# Input:  x [M, K] with K even; K = 8192, M = 8192
# Output: out [M, K//2] = [M, 4096]
#
# One row per program.  BLOCK_SIZE = N = 4096 exactly matches the chunk
# dimension, so masks are unnecessary.  Added memory-linearity hints
# (multiple_of, max_contiguous) to help the compiler generate widened
# vector loads/stores.
# ---------------------------------------------------------------------------

@triton.jit
def swiglu_kernel(
    x_ptr,                # pointer to input (2D flattened)
    out_ptr,              # pointer to output (2D flattened)
    stride_x_row,         # row stride of x (in elements) = K
    stride_out_row,       # row stride of out (in elements) = N
    BLOCK_SIZE: tl.constexpr,   # ≡ N = 4096
):
    # ---- 1. row index ----------------------------------------------------
    row = tl.program_id(0)

    # ---- 2. pointers for this row ----------------------------------------
    x_row = x_ptr + row * stride_x_row
    out_row = out_ptr + row * stride_out_row

    # ---- 3. column indices (no mask needed) ------------------------------
    col_offs = tl.arange(0, BLOCK_SIZE)

    # ---- 4. load a and b with alignment hints ----------------------------
    # Guarantee that the base pointer is aligned to 16 elements (64 bytes)
    # and that each loaded region is contiguous within the row.
    a_ptr = tl.multiple_of(x_row + col_offs, 16)
    a_ptr = tl.max_contiguous(a_ptr, BLOCK_SIZE)
    a = tl.load(a_ptr, eviction_policy='evict_first')

    b_ptr = tl.multiple_of(x_row + col_offs + BLOCK_SIZE, 16)
    b_ptr = tl.max_contiguous(b_ptr, BLOCK_SIZE)
    b = tl.load(b_ptr, eviction_policy='evict_first')

    # ---- 5. compute SiLU(a) = a * sigmoid(a) -----------------------------
    sig_a = tl.sigmoid(a)
    silu_a = a * sig_a

    # ---- 6. gating: silu(a) * b ------------------------------------------
    out = silu_a * b

    # ---- 7. store result, also hint the output pointer --------------------
    out_ptr_row = tl.multiple_of(out_row + col_offs, 16)
    out_ptr_row = tl.max_contiguous(out_ptr_row, BLOCK_SIZE)
    tl.store(out_ptr_row, out, eviction_policy='evict_first')


# ---------------------------------------------------------------------------
# Public entry point – allocates output and launches the kernel.
# ---------------------------------------------------------------------------

def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    x: float32 tensor of shape [M, K]  (K even)
    returns: float32 tensor of shape [M, K//2] (SWiGLU activation)
    """
    M, K = x.shape
    N = K // 2                          # output feature dimension

    # Allocate output
    out = torch.empty(M, N, device=x.device, dtype=torch.float32)

    # Launch configuration
    BLOCK_SIZE = N                      # 4096, exactly covers one row's a/b
    grid = (M,)                         # one program per row

    swiglu_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,                    # good balance occupancy vs. registers
        num_stages=4,                   # moderate pipelining
    )
    return out