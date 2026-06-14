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
    # One program per row
    row = tl.program_id(0)
    x_row_ptr = x_ptr + row * N
    out_row_ptr = out_ptr + row * N

    # Contiguous columns, no mask needed since N == BLOCK
    offsets = tl.arange(0, BLOCK)
    x = tl.load(x_row_ptr + offsets, mask=None, eviction_policy='evict_first')

    # Compute RMS: sqrt(mean(x^2) + eps)
    x_sq = x * x
    sum_sq = tl.sum(x_sq, axis=0)   # reduction across the block
    mean_sq = sum_sq / N
    rstd = tl.rsqrt(mean_sq + eps)

    out = x * rstd
    tl.store(out_row_ptr + offsets, out, mask=None, eviction_policy='evict_last')


def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous()
    M, N = x.shape   # 8192, 4096
    out = torch.empty_like(x)

    BLOCK = N         # N = 4096 → exactly one block per row
    grid = (M,)

    # Increase warps to 32 (1024 threads) to better hide memory latency
    # and raise DRAM bandwidth utilisation closer to peak.
    rms_norm_kernel[grid](
        x, out,
        N=N,
        eps=1e-5,
        BLOCK=BLOCK,
        num_warps=32,
        num_stages=1,
    )
    return out