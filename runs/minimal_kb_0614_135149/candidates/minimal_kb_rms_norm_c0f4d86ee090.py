import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    x_ptr,
    out_ptr,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Kernel for RMS normalization: out = x * rsqrt(mean(x^2, dim=-1) + eps)
    One program per row, BLOCK == N.
    """
    row = tl.program_id(0)
    x_row_ptr = x_ptr + row * N
    out_row_ptr = out_ptr + row * N

    offsets = tl.arange(0, BLOCK)
    # Load entire row – streaming data, hint to evict early
    x = tl.load(x_row_ptr + offsets, mask=None, eviction_policy="evict_first")

    # Single‑pass RMS computation
    x_sq = x * x
    sum_sq = tl.sum(x_sq, axis=0)          # sum over the row
    mean_sq = sum_sq / N
    rstd = tl.rsqrt(mean_sq + eps)

    out = x * rstd

    # Store result – mark as evict_last for subsequent consumers
    tl.store(out_row_ptr + offsets, out, mask=None, eviction_policy="evict_last")


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    Apply RMS normalization along the last dimension (f32[8192, 4096] -> f32[8192, 4096]).
    x must be contiguous.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    M, N = x.shape

    # For the given shape N=4096, one tile covers the entire row – no mask needed.
    BLOCK = N
    out = torch.empty_like(x)

    # Launch one thread block per row
    grid = (M,)

    rms_norm_kernel[grid](
        x,
        out,
        N=N,
        eps=1e-5,
        BLOCK=BLOCK,
        num_warps=16,   # increased from 8 to 16 to boost memory‑level parallelism
        num_stages=1,   # no software pipeline needed — single‑step load/compute/store
    )

    return out