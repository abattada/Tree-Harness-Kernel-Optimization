import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    x_ptr,
    out_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    One program per row.  Computes softmax over the last dimension.
    """
    row = tl.program_id(0)
    x_row_ptr = x_ptr + row * N
    out_row_ptr = out_ptr + row * N

    offsets = tl.arange(0, BLOCK)
    # Load the entire row (BLOCK == N, so mask is trivial, but keep it safe)
    x = tl.load(x_row_ptr + offsets, mask=offsets < N)

    # Stable softmax: subtract row max before exp
    row_max = tl.max(x, axis=0)
    ex = tl.exp(x - row_max)
    row_sum = tl.sum(ex, axis=0)
    out = ex / row_sum

    tl.store(out_row_ptr + offsets, out, mask=offsets < N)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous()
    M, N = x.shape
    out = torch.empty_like(x)

    BLOCK = triton.next_power_of_2(N)  # 4096, already a power of two
    grid = (M,)

    softmax_kernel[grid](
        x,
        out,
        N=N,
        BLOCK=BLOCK,
        num_warps=8,      # good for a BLOCK of 4096 floats
        num_stages=1,     # no pipelining needed for this simple reduction
    )
    return out