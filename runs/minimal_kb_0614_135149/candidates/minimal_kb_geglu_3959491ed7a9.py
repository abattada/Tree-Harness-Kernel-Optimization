import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def geglu_kernel(
    x_ptr,
    out_ptr,
    stride_x,
    stride_out,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    One program per row; each program processes the entire output row
    (N elements) in a single block.  BLOCK_SIZE must equal N to avoid masking.
    """
    tl.static_assert(BLOCK_SIZE == N, "BLOCK_SIZE must equal N for this specialisation")

    # Row index
    row = tl.program_id(0)

    # Column indices – every element is valid because BLOCK_SIZE == N
    col_offs = tl.arange(0, BLOCK_SIZE)

    # Base pointers for this row
    x_row = x_ptr + row * stride_x
    out_row = out_ptr + row * stride_out

    # Load first half (a) and second half (b) with streaming hints
    a = tl.load(x_row + col_offs, cache_modifier=".cg", eviction_policy="evict_first")
    b = tl.load(x_row + col_offs + N, cache_modifier=".cg", eviction_policy="evict_first")

    # Compute GELU with tanh approximation:
    # GELU(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
    sqrt2_over_pi = 0.7978845608028654
    c = 0.044715
    a3 = a * a * a
    inner = sqrt2_over_pi * (a + c * a3)
    tanh_val = libdevice.tanh(inner)
    gelu_a = 0.5 * a * (1.0 + tanh_val)

    # Gate
    out = gelu_a * b

    # Store result – also evict-first as it is written once and not reused
    tl.store(out_row + col_offs, out, eviction_policy="evict_first")


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    x: float32 tensor of shape [M, K] with K even (here 8192, 8192)
    returns: float32 tensor of shape [M, K//2] = [8192, 4096]
    """
    M, K = x.shape
    N = K // 2                             # output feature dimension (4096)

    out = torch.empty(M, N, device=x.device, dtype=torch.float32)

    grid = (M,)                            # one program per row
    geglu_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        N=N,
        BLOCK_SIZE=4096,                  # full output row
        num_warps=8,                      # good occupancy / register balance
        num_stages=4,                     # moderate pipelining
    )
    return out