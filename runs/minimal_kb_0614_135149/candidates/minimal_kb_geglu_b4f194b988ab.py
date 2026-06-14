import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def geglu_kernel(
    x_ptr,                # shape [M, K] flattened
    out_ptr,              # shape [M, N] flattened
    stride_x,             # stride between rows of x (elements)
    stride_out,           # stride between rows of out (elements)
    BLOCK_SIZE: tl.constexpr,   # exactly N (the full output row)
    N: tl.constexpr,            # output width = K//2
):
    # Safety: this kernel is only used when BLOCK_SIZE == N,
    # so there are no partial tiles and we can drop all masks.
    tl.static_assert(BLOCK_SIZE == N, "BLOCK_SIZE must equal N")

    # Row index
    row = tl.program_id(0)

    # Column offsets for the whole output row
    col_offs = tl.arange(0, BLOCK_SIZE)

    # Base pointers for this row
    x_row = x_ptr + row * stride_x
    out_row = out_ptr + row * stride_out

    # Load the two halves of the input.
    # cache_modifier=".cg" uses the streaming (.cg) cache operator to bypass L1
    # (the data is read exactly once), and evict_first discards the cache line
    # as soon as it is consumed, keeping cache pollution minimal.
    a = tl.load(x_row + col_offs, cache_modifier=".cg", eviction_policy='evict_first')
    b = tl.load(x_row + col_offs + N, cache_modifier=".cg", eviction_policy='evict_first')

    # GELU with tanh approximation:
    #   GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt2_over_pi = 0.7978845608028654
    c = 0.044715

    a3 = a * a * a
    inner = sqrt2_over_pi * (a + c * a3)
    tanh_val = libdevice.tanh(inner)
    gelu_a = 0.5 * a * (1.0 + tanh_val)

    # Gate
    out = gelu_a * b

    # Write result (also evict_first since it is a final write, not read again)
    tl.store(out_row + col_offs, out, eviction_policy='evict_first')


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    x: float32 tensor of shape [8192, 8192]
    returns: float32 tensor of shape [8192, 4096]
    """
    M, K = x.shape
    N = K // 2                     # 4096

    # Allocate output
    out = torch.empty(M, N, device=x.device, dtype=torch.float32)

    # One program per row; BLOCK_SIZE covers the entire row → no masking
    BLOCK_SIZE = N
    grid = (M,)

    # Launch with more warps to increase memory-level parallelism and fewer
    # stages to reduce register pressure with the larger block (512 threads).
    geglu_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        N=N,
        num_warps=16,
        num_stages=2,
    )

    return out