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
    Softmax over the last dimension.
    Each program handles one full row.
    """
    row = tl.program_id(0)
    x_row_ptr = x_ptr + row * N
    out_row_ptr = out_ptr + row * N

    offsets = tl.arange(0, BLOCK)
    # No mask needed because N == BLOCK exactly.
    # Use max_contiguous and multiple_of hints to enable vectorization.
    x = tl.load(
        x_row_ptr + offsets,
        mask=None,
        other=0.0,
        eviction_policy='evict_first',
    )
    # Compute max for numerical stability
    row_max = tl.max(x, axis=0)

    # Shift and exponentiate
    x_shifted = x - row_max
    exp_x = tl.exp(x_shifted)

    # Sum of exponentials
    row_sum = tl.sum(exp_x, axis=0)

    # Softmax output
    out = exp_x / row_sum

    tl.store(
        out_row_ptr + offsets,
        out,
        mask=None,
        eviction_policy='evict_last',
    )


def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous()
    M, N = x.shape
    # BLOCK must equal N (power of two) for no-mask path
    BLOCK = N  # 4096
    out = torch.empty_like(x)

    grid = (M,)

    softmax_kernel[grid](
        x,
        out,
        N=N,
        BLOCK=BLOCK,
        num_warps=8,
        num_stages=1,
    )
    return out