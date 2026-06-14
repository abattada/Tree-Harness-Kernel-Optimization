import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ---------------------------------------------------------------------------
# GeGLU kernel – gated GELU with tanh approximation.
# Specialized for the exact shapes M=8192, K=8192, N=4096.
# Constexpr dimensions enable more aggressive compiler optimizations.
# ---------------------------------------------------------------------------

@triton.jit
def geglu_kernel(
    x_ptr,                # pointer to input
    out_ptr,              # pointer to output
    K: tl.constexpr,      # input feature dim (= 2*N)
    N: tl.constexpr,      # output feature dim
    BLOCK_SIZE: tl.constexpr,   # must equal N
):
    # Only one block per row, covering the whole output row.
    tl.static_assert(BLOCK_SIZE == N,
                     "BLOCK_SIZE must equal N for this specialization")
    tl.static_assert(K == 2 * N, "K must be 2 * N")

    row = tl.program_id(0)

    # Column offsets – no mask needed because BLOCK_SIZE == N exactly.
    col_offs = tl.arange(0, BLOCK_SIZE)

    # Row pointers (strides are constexpr: stride_x = K, stride_out = N)
    x_row = x_ptr + row * K
    out_row = out_ptr + row * N

    # Load a (first half) and b (second half)
    a = tl.load(x_row + col_offs)
    b = tl.load(x_row + col_offs + N)

    # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt2_over_pi = 0.7978845608028654
    c = 0.044715

    a3 = a * a * a
    inner = sqrt2_over_pi * (a + c * a3)
    tanh_val = libdevice.tanh(inner)
    gelu_a = 0.5 * a * (1.0 + tanh_val)

    out = gelu_a * b

    tl.store(out_row + col_offs, out)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    x: float32 tensor of shape [8192, 8192]
    returns: float32 tensor of shape [8192, 4096]
    """
    M, K = x.shape
    N = K // 2

    out = torch.empty(M, N, device=x.device, dtype=torch.float32)

    BLOCK_SIZE = N          # covers the whole output row
    grid = (M,)

    # Use more warps to hide memory latency, fewer stages (no shared memory reuse)
    geglu_kernel[grid](
        x, out,
        K=K, N=N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=16,
        num_stages=2,
    )

    return out