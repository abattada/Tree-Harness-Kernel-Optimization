import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel: SWiGLU (persistent kernel version)
# Input:  x [M, K] with K even; K = 8192, M = 8192
# Output: out [M, K//2] = [M, 4096]
#
# Instead of launching one program per row (8192), we use a persistent
# kernel with a fixed grid size (e.g., 256 programs). Each program loops
# over rows in a grid‑stride fashion. This reduces launch overhead and
# improves cache locality because consecutive rows are handled by the
# same SM, while keeping memory requests coalesced.
# ---------------------------------------------------------------------------

@triton.jit
def swiglu_kernel(
    x_ptr,                # pointer to input (2D flattened)
    out_ptr,              # pointer to output (2D flattened)
    stride_x_row,         # row stride of x (in elements) = K
    stride_out_row,       # row stride of out (in elements) = N
    M,                    # total number of rows
    BLOCK_SIZE: tl.constexpr,   # ≡ N = 4096
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)   # persistent grid size

    row = pid
    while row < M:
        # pointers for this row
        x_row = x_ptr + row * stride_x_row
        out_row = out_ptr + row * stride_out_row

        # column indices (no mask: BLOCK_SIZE == N exactly)
        col_offs = tl.arange(0, BLOCK_SIZE)

        # load a and b (contiguous halves of the row)
        a = tl.load(x_row + col_offs, eviction_policy='evict_first')
        b = tl.load(x_row + col_offs + BLOCK_SIZE, eviction_policy='evict_first')

        # SiLU(a) = a * sigmoid(a)
        silu_a = a * tl.sigmoid(a)

        # gating: silu(a) * b
        out = silu_a * b

        # store result
        tl.store(out_row + col_offs, out, eviction_policy='evict_first')

        # next row
        row += num_programs


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

    # Persistent grid size – use a small multiple of SMs (e.g., 256)
    # For M = 8192 this gives 8192/256 = 32 iterations per program.
    PERSISTENT_GRID = 256
    grid = (min(M, PERSISTENT_GRID),)

    swiglu_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        M,
        BLOCK_SIZE=N,
        num_warps=8,
        num_stages=4,
    )
    return out