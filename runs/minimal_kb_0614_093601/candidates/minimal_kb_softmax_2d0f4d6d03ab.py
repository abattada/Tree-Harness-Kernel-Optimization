import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    x_ptr, out_ptr,
    stride_x, stride_out,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_row_ptr = x_ptr + row * stride_x
    out_row_ptr = out_ptr + row * stride_out

    cols = tl.arange(0, BLOCK_SIZE)
    # N == BLOCK_SIZE, so no mask needed
    x = tl.load(x_row_ptr + cols)

    # Compute max for numerical stability
    row_max = tl.max(x, axis=0)

    # Compute exp(x - max) and sum
    shifted = x - row_max
    exp_shifted = tl.exp(shifted)
    sum_exp = tl.sum(exp_shifted, axis=0)

    # Normalize
    out = exp_shifted / sum_exp
    tl.store(out_row_ptr + cols, out)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous()
    M, N = x.shape
    out = torch.empty_like(x)
    grid = (M,)
    BLOCK_SIZE = triton.next_power_of_2(N)  # =4096
    softmax_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,      # good for BLOCK_SIZE=4096 reduction
        num_stages=1,     # no pipelining needed
    )
    return out