import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


# ---------------------------------------------------------------------------
# Triton kernel: GeGLU
# Input:  x  [M, K]  with K even  (M = 8192, K = 8192)
# Output: out [M, K//2]            (N = 4096)
# One program per row, full row inside one block  ->  no masking needed.
#
# Improvement over parent: added vectorization hints (tl.max_contiguous,
# tl.multiple_of) to let the compiler issue wider, coalesced loads/stores
# and potentially increase bandwidth utilisation towards 90 % peak.
# ---------------------------------------------------------------------------

@triton.jit
def geglu_kernel(
    x_ptr,
    out_ptr,
    stride_x_row,
    stride_out_row,
    BLOCK_SIZE: tl.constexpr,
    N: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE == N,
                     "BLOCK_SIZE must equal N for this specialization")

    # Row index
    row = tl.program_id(0)

    # Column offsets (always valid, no mask)
    col_offs = tl.arange(0, BLOCK_SIZE)

    # Base pointers for this row
    x_row  = x_ptr  + row * stride_x_row
    out_row = out_ptr + row * stride_out_row

    # --- Load a (first half) and b (second half) with alignment hints ---
    a_ptrs = tl.max_contiguous(tl.multiple_of(x_row + col_offs, BLOCK_SIZE),
                               BLOCK_SIZE)
    b_ptrs = tl.max_contiguous(tl.multiple_of(x_row + col_offs + N, BLOCK_SIZE),
                               BLOCK_SIZE)

    a = tl.load(a_ptrs, eviction_policy='evict_first')
    b = tl.load(b_ptrs, eviction_policy='evict_first')

    # --- GELU with tanh approximation ---
    sqrt2_over_pi = 0.7978845608028654
    c = 0.044715

    a3    = a * a * a
    inner = sqrt2_over_pi * (a + c * a3)
    tanh_val = libdevice.tanh(inner)
    gelu_a  = 0.5 * a * (1.0 + tanh_val)

    # --- Gating ---
    out = gelu_a * b

    # --- Store output with alignment hints ---
    out_ptrs = tl.max_contiguous(tl.multiple_of(out_row + col_offs, BLOCK_SIZE),
                                 BLOCK_SIZE)
    tl.store(out_ptrs, out, eviction_policy='evict_first')


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

    # Allocate output
    out = torch.empty(M, N, device=x.device, dtype=torch.float32)

    BLOCK_SIZE = N          # always 4096 for the given shapes
    grid = (M,)             # one program per row

    geglu_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        N=N,
        num_warps=8,
        num_stages=4,
    )

    return out