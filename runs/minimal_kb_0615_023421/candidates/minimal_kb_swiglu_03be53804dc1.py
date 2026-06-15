import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel: SwiGLU (silu gating)
# Input:  x [M, K]  with K even, e.g. K = 8192
# Output: out [M, K//2] = silu(a) * b  where a,b = x.chunk(2, -1)
#
# Optimizations:
#  - Full-row blocks (BLOCK_SIZE = N) → no masking overhead.
#  - tl.max_contiguous hints on column offsets → enable wider vector loads.
#  - tl.multiple_of hints on row pointers → 128-byte alignment.
#  - eviction_policy='evict_first' for both loads and store (streaming).
#  - 8 warps / 4 stages: good balance for large-row occupancy.
# ---------------------------------------------------------------------------

@triton.jit
def swiglu_kernel(
    x_ptr,                # pointer to input (2D, row-major)
    out_ptr,              # pointer to output (2D, row-major)
    stride_x_row,         # row stride of x (in elements) = K
    stride_out_row,       # row stride of out (in elements) = N
    BLOCK_SIZE: tl.constexpr,   # = N = K//2
):
    row = tl.program_id(0)

    # Row base pointers with alignment hint (32 floats = 128 bytes)
    x_row = tl.multiple_of(x_ptr + row * stride_x_row, 32)
    out_row = tl.multiple_of(out_ptr + row * stride_out_row, 32)

    # Contiguous column offsets for one half of the row
    col_offs = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)

    # Load a and b from the two halves (evict_first – only read once)
    a = tl.load(x_row + col_offs, eviction_policy='evict_first')
    b = tl.load(x_row + col_offs + BLOCK_SIZE, eviction_policy='evict_first')

    # silu(x) = x * sigmoid(x)
    silu_a = a * tl.sigmoid(a)

    # Gating
    out = silu_a * b

    # Store result (evict_first – never reused)
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

    out = torch.empty(M, N, device=x.device, dtype=torch.float32)

    # One program per row, full-row block
    swiglu_kernel[(M,)](
        x, out,
        x.stride(0), out.stride(0),
        BLOCK_SIZE=N,
        num_warps=8,
        num_stages=4,
    )
    return out