import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel: SwiGLU (silu gating) with multirow-per-program loop
# Input:  x [M, K]  with K even, e.g. K = 8192
# Output: out [M, K//2] = silu(a) * b  where a,b = x.chunk(2, -1)
#
# This version processes multiple rows in each program to reduce launch
# overhead and improve SM utilization.  Full-row blocks avoid masking.
# ---------------------------------------------------------------------------

@triton.jit
def swiglu_kernel(
    x_ptr,                # pointer to input (2D, row-major)
    out_ptr,              # pointer to output (2D, row-major)
    stride_x_row,         # row stride of x (in elements) = K
    stride_out_row,       # row stride of out (in elements) = N
    M,                    # total number of rows (for boundary)
    BLOCK_SIZE: tl.constexpr,   # = N = K//2
    ROWS_PER_PROG: tl.constexpr, # number of rows handled by one program
):
    # ---- 1. program id and row range ------------------------------------
    pid = tl.program_id(0)
    start_row = pid * ROWS_PER_PROG
    end_row = start_row + ROWS_PER_PROG
    # Handle boundary (assumes M is multiple of ROWS_PER_PROG, but safe)
    if end_row > M:
        end_row = M

    # ---- 2. column offsets (constant across rows) ----------------------
    col_offs = tl.arange(0, BLOCK_SIZE)

    # ---- 3. loop over rows within this program -------------------------
    for row in range(start_row, end_row):
        # Pointers for this row
        x_row = x_ptr + row * stride_x_row
        out_row = out_ptr + row * stride_out_row

        # Load a and b (both halves of the row)
        a = tl.load(x_row + col_offs, eviction_policy='evict_first')
        b = tl.load(x_row + col_offs + BLOCK_SIZE, eviction_policy='evict_first')

        # silu(x) = x * sigmoid(x)
        sig_a = tl.sigmoid(a)
        silu_a = a * sig_a

        # Gating
        out = silu_a * b

        # Store result
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

    # Choose rows per program to keep grid size moderate
    # 8 rows per program gives grid = 8192//8 = 1024
    ROWS_PER_PROG = 8

    # Launch configuration
    BLOCK_SIZE = N                      # 4096, exactly covers one row
    grid = (triton.cdiv(M, ROWS_PER_PROG),)   # handle boundary cleanly

    swiglu_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        M,
        BLOCK_SIZE=BLOCK_SIZE,
        ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=8,                     # good balance for this problem
        num_stages=4,                    # moderate pipeline depth
    )
    return out