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
    # One program per row; BLOCK == N, so no boundary mask needed.
    row = tl.program_id(0)
    x_row_ptr = x_ptr + row * N
    out_row_ptr = out_ptr + row * N

    offsets = tl.arange(0, BLOCK)
    x = tl.load(x_row_ptr + offsets, mask=None, eviction_policy='evict_first')

    x_sq = x * x
    sum_sq = tl.sum(x_sq, axis=0)       # reduce within the block
    mean_sq = sum_sq / N                 # N is constexpr → compile-time reciprocal
    rstd = tl.rsqrt(mean_sq + eps)

    out = x * rstd
    tl.store(out_row_ptr + offsets, out, mask=None, eviction_policy='evict_last')


def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous()
    M, N = x.shape          # 8192, 4096
    out = torch.empty_like(x)

    BLOCK = N               # exactly one tile per row
    grid = (M,)

    rms_norm_kernel[grid](
        x, out,
        N=N,
        eps=1e-5,
        BLOCK=BLOCK,
        num_warps=16,       # increased from 8 to boost parallelism and reduce per‑thread work
        num_stages=1,
    )
    return out