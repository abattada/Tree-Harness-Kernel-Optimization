import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    x_ptr, out_ptr,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)

    # Compute row pointers
    x_row_ptr = x_ptr + row * N
    out_row_ptr = out_ptr + row * N

    # Alignment hints for vectorized memory accesses
    tl.multiple_of(x_row_ptr, 16)
    tl.multiple_of(out_row_ptr, 16)
    tl.multiple_of(offsets, 16)

    # Load the entire row with streaming policy
    x = tl.load(x_row_ptr + offsets, mask=None, eviction_policy='evict_first')

    # Single‑pass RMS computation
    x_sq = x * x
    sum_sq = tl.sum(x_sq, axis=0)       # parallel reduction inside the block
    mean_sq = sum_sq / N                # N is constexpr; compiler optimises division
    rstd = tl.rsqrt(mean_sq + eps)

    out = x * rstd

    # Store the result with reuse‑friendly eviction policy
    tl.store(out_row_ptr + offsets, out, mask=None, eviction_policy='evict_last')


def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous()
    M, N = x.shape  # 8192, 4096
    out = torch.empty_like(x)

    BLOCK = N               # One tile covers the whole row
    grid = (M,)

    rms_norm_kernel[grid](
        x, out,
        N=N,
        eps=1e-5,
        BLOCK=BLOCK,
        num_warps=16,        # More warps to hide memory latency
        num_stages=1,        # No pipelining needed (single load/store per row)
        fast_math=True,      # Approximate rsqrt reduces latency vs. full‑precision
    )
    return out