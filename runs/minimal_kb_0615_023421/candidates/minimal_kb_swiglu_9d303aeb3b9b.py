import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel: SWiGLU (multirow per program)
# Input:  x [M, K] with K even; M=8192, K=8192
# Output: out [M, K//2] = [M, 4096]
#
# Processing multiple rows per program reduces launch overhead and improves
# occupancy on large row counts.  We use ROWS_PER_PROG=4, and a simple loop.
# Because ROWS_PER_PROG divides M exactly, no boundary handling is needed.
# ---------------------------------------------------------------------------

@triton.jit
def swiglu_kernel(
    x_ptr,
    out_ptr,
    stride_x_row,
    stride_out_row,
    BLOCK_SIZE: tl.constexpr,          # 4096
    ROWS_PER_PROG: tl.constexpr,       # 4
):
    pid = tl.program_id(0)
    base_row = pid * ROWS_PER_PROG

    # column offsets (full tile, no mask)
    col_offs = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)

    # loop over ROWS_PER_PROG rows
    for i in range(ROWS_PER_PROG):
        row = base_row + i
        x_row = x_ptr + row * stride_x_row
        out_row = out_ptr + row * stride_out_row

        # load a and b
        a = tl.load(x_row + col_offs, eviction_policy='evict_first')
        b = tl.load(x_row + col_offs + BLOCK_SIZE, eviction_policy='evict_first')

        # SiLU(a) = a * sigmoid(a)
        silu_a = a * tl.sigmoid(a)

        # gated product
        out = silu_a * b

        # store result
        tl.store(out_row + col_offs, out, eviction_policy='evict_first')


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
    ROWS_PER_PROG = 4                   # tunable – reduces grid size
    assert M % ROWS_PER_PROG == 0, "M must be divisible by ROWS_PER_PROG"
    grid = (M // ROWS_PER_PROG,)

    swiglu_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=8,                    # balanced occupancy / registers
        num_stages=4,                   # moderate pipelining
    )
    return out