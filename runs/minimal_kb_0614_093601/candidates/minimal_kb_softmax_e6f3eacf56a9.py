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
    row = tl.program_id(0)
    x_row = x_ptr + row * N
    out_row = out_ptr + row * N

    offsets = tl.arange(0, BLOCK)
    # Load the whole row (no mask since N == BLOCK)
    x = tl.load(x_row + offsets)

    # Compute row-wise max
    row_max = tl.max(x, axis=0)

    # Compute exp(x - max) and sum
    shifted = x - row_max
    exp_shifted = tl.exp(shifted)
    sum_exp = tl.sum(exp_shifted, axis=0)

    # Normalize
    y = exp_shifted / sum_exp

    # Store result
    tl.store(out_row + offsets, y)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous()
    M, N = x.shape
    out = torch.empty_like(x)

    BLOCK = triton.next_power_of_2(N)  # =4096
    grid = (M,)
    softmax_kernel[grid](
        x,
        out,
        N=N,
        BLOCK=BLOCK,
        num_warps=8,      # good for BLOCK=4096
        num_stages=1,     # no pipelining needed
    )
    return out