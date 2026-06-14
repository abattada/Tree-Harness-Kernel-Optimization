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
    # Program ID along rows
    row = tl.program_id(0)
    # Base pointers for this row
    x_row_ptr = x_ptr + row * N
    out_row_ptr = out_ptr + row * N

    # Offsets for this block (full row)
    offsets = tl.arange(0, BLOCK)
    # Load input row – no mask needed since N == BLOCK
    # Use eviction_policy='evict_first' because the data is only read once
    x = tl.load(x_row_ptr + offsets, eviction_policy='evict_first')

    # Compute max over the row for numerical stability
    row_max = tl.max(x, axis=0)

    # Compute exp(x - max) and sum
    x_shifted = x - row_max
    exp_x = tl.exp(x_shifted)
    row_sum = tl.sum(exp_x, axis=0)

    # Softmax output
    out = exp_x / row_sum

    # Store – also evict_first because this is a streaming write
    tl.store(out_row_ptr + offsets, out, eviction_policy='evict_first')


def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous()
    M, N = x.shape
    out = torch.empty_like(x)

    # Block size equals N (4096), power of two, so no masking needed
    BLOCK = N  # 4096

    grid = (M,)

    softmax_kernel[grid](
        x,
        out,
        N=N,
        BLOCK=BLOCK,
        num_warps=8,       # good for reduction over 4096 elements
        num_stages=1,      # no pipelining needed
    )
    return out