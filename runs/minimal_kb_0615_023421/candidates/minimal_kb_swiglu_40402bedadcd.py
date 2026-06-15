import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel: SWiGLU (grid‑stride loop over rows)
# Input:  x [M, K] with K even; M = 8192, K = 8192
# Output: out [M, K//2] = [M, 4096]
#
# Each program processes ROWS_PER_PROG rows sequentially to amortize
# launch overhead and improve occupancy.  Masks are dropped because
# M is a multiple of ROWS_PER_PROG and BLOCK_SIZE == N = 4096.
# ---------------------------------------------------------------------------

ROWS_PER_PROG = 8                        # tunable – 8 rows per program

@triton.jit
def swiglu_kernel(
    x_ptr,                # pointer to input (2D flattened)
    out_ptr,              # pointer to output (2D flattened)
    stride_x_row,         # row stride of x (in elements) = K
    stride_out_row,       # row stride of out (in elements) = N
    BLOCK_SIZE: tl.constexpr,   # ≡ N = 4096
    ROWS_PER_PROG_: tl.constexpr,   # number of rows per program
):
    # ---- 1. program id and base row index --------------------------------
    pid = tl.program_id(0)
    base_row = pid * ROWS_PER_PROG_

    # ---- 2. column offsets (no mask needed) -----------------------------
    col_offs = tl.arange(0, BLOCK_SIZE)

    # ---- 3. loop over rows assigned to this program ----------------------
    for i in range(ROWS_PER_PROG_):
        row = base_row + i

        # pointers for this row
        x_row = x_ptr + row * stride_x_row
        out_row = out_ptr + row * stride_out_row

        # load a and b (contiguous halves of the row)
        a = tl.load(x_row + col_offs, eviction_policy='evict_first')
        b = tl.load(x_row + col_offs + BLOCK_SIZE, eviction_policy='evict_first')

        # SiLU(a) = a * sigmoid(a)
        sig_a = tl.sigmoid(a)
        silu_a = a * sig_a

        # gating: silu(a) * b
        out = silu_a * b

        # store result
        tl.store(out_row + col_offs, out, eviction_policy='evict_first')


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    x: float32 tensor of shape [M, K]  (K even)
    returns: float32 tensor of shape [M, K//2]
    """
    M, K = x.shape
    N = K // 2

    out = torch.empty(M, N, device=x.device, dtype=torch.float32)

    # Grid launch: each program handles ROWS_PER_PROG rows
    assert M % ROWS_PER_PROG == 0, "M must be a multiple of ROWS_PER_PROG"
    grid = (M // ROWS_PER_PROG,)

    swiglu_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        BLOCK_SIZE=N,
        ROWS_PER_PROG_=ROWS_PER_PROG,
        num_warps=8,
        num_stages=4,
    )
    return out