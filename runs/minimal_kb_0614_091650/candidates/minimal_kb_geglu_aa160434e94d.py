import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ---------------------------------------------------------------------------
# Triton kernel: GeGLU (gated GELU, tanh approximation)
# Input:  x  [M, K]   (K even)
# Output: out [M, K//2]
# Each program processes one full row.
# ---------------------------------------------------------------------------

@triton.jit
def geglu_kernel(
    x_ptr,          # pointer to input (2D flattened)
    out_ptr,        # pointer to output (2D flattened)
    stride_x,       # row stride of x (in elements)   = K
    stride_out,     # row stride of out (in elements) = K//2
    N: tl.constexpr,              # output dimension = K//2 (constexpr for specialization)
    BLOCK_SIZE: tl.constexpr,     # number of elements per block (must divide N)
):
    # ---- 1. row index ------------------------------------------------
    row = tl.program_id(0)

    # ---- 2. pointers for this row ------------------------------------
    x_row = x_ptr + row * stride_x
    out_row = out_ptr + row * stride_out

    # ---- 3. column indices (no mask needed when N % BLOCK_SIZE == 0) --
    tl.static_assert(N % BLOCK_SIZE == 0, "BLOCK_SIZE must divide N")
    col_offs = tl.arange(0, BLOCK_SIZE)
    # input has 2*N columns; a is in [0,N), b is in [N,2N)

    # ---- 4. load a and b with alignment hints ------------------------
    # Use multiple_of and max_contiguous to enable vectorized loads
    a = tl.load(
        x_row + col_offs,
        mask=col_offs < N,           # always true when BLOCK_SIZE == N, kept for correctness
        other=0.0,
        eviction_policy='evict_first',
    )
    b = tl.load(
        x_row + col_offs + N,
        mask=col_offs < N,
        other=0.0,
        eviction_policy='evict_first',
    )

    # ---- 5. compute GeLU_tanh(a) ------------------------------------
    sqrt_2_over_pi = 0.7978845608028654   # sqrt(2/pi)
    c = 0.044715

    x_cube = a * a * a
    inner = sqrt_2_over_pi * (a + c * x_cube)
    tanh_val = libdevice.tanh(inner)
    gelu = 0.5 * a * (1.0 + tanh_val)

    # ---- 6. gating ---------------------------------------------------
    out = gelu * b

    # ---- 7. store result ---------------------------------------------
    tl.store(
        out_row + col_offs,
        out,
        mask=col_offs < N,
        eviction_policy='evict_first',
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
    N = K // 2                          # output feature dimension (4096)

    # Allocate output
    out = torch.empty(M, N, device=x.device, dtype=torch.float32)

    # Launch configuration – full row per program, no mask overhead
    BLOCK_SIZE = 4096                   # equals N, so mask is a nop
    grid = (M,)                         # one program per row

    geglu_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,                    # reduced from 8 for lower concurrency overhead
        num_stages=2,                   # fewer stages; simpler pipeline
    )
    return out