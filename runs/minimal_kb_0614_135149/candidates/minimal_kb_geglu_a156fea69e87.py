import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ---------------------------------------------------------------------------
# Triton kernel: GeGLU (gated GELU with tanh approximation)
# Input:  x [M, K]  with K even, K = 8192
# Output: out [M, K//2] = [M, 4096]
#
# One program per row, full-row loads/stores, no boundary masks needed.
# Constant specialization hints and cache eviction policies are applied.
# ---------------------------------------------------------------------------

@triton.jit
def geglu_kernel(
    x_ptr,                     # pointer to input (2D flattened)
    out_ptr,                   # pointer to output (2D flattened)
    stride_x,                  # row stride of x (in elements) = K
    stride_out,                # row stride of out (in elements) = N
    BLOCK_SIZE: tl.constexpr,  # output dimension N = 4096 (full row)
    N: tl.constexpr,           # N = K // 2
):
    # Guarantee that one block covers exactly one output row
    tl.static_assert(BLOCK_SIZE == N,
                     "BLOCK_SIZE must equal N for mask-free execution")

    # Row index
    row = tl.program_id(0)

    # Column indices (always within bounds: BLOCK_SIZE == N)
    col_offs = tl.arange(0, BLOCK_SIZE)

    # Base pointers for this row
    x_base = x_ptr + row * stride_x
    out_base = out_ptr + row * stride_out

    # Load first half (A) and second half (B) with streaming hints
    a = tl.load(x_base + col_offs,          eviction_policy="evict_first")
    b = tl.load(x_base + col_offs + N,      eviction_policy="evict_first")

    # ----- GELU with tanh approximation --------------------------------------
    # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    sqrt2_over_pi = 0.7978845608028654
    c = 0.044715

    a3 = a * a * a
    inner = sqrt2_over_pi * (a + c * a3)
    tanh_val = libdevice.tanh(inner)
    gelu_a = 0.5 * a * (1.0 + tanh_val)

    # Gate: gelu(a) * b
    out = gelu_a * b

    # Store result (streaming, won't be read again)
    tl.store(out_base + col_offs, out, eviction_policy="evict_first")


# ---------------------------------------------------------------------------
# Public entry point – allocates output and launches the kernel.
# ---------------------------------------------------------------------------

def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    x: float32 tensor of shape [M, K] with K even (expects [8192, 8192])
    returns: float32 tensor of shape [M, K//2] = [8192, 4096]
    """
    M, K = x.shape
    N = K // 2                          # 4096

    # Allocate output (same device/dtype as input)
    out = torch.empty(M, N, device=x.device, dtype=torch.float32)

    # Launch configuration: one program per row
    BLOCK_SIZE = 4096                   # equals N, covers the whole row
    grid = (M,)                         # 8192 programs

    geglu_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        N=N,
        num_warps=8,                    # balances register usage / occupancy
        num_stages=4,                   # moderate prefetching
    )

    return out