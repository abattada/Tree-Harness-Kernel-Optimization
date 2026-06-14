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
    x_row_ptr = x_ptr + row * N
    out_row_ptr = out_ptr + row * N

    # Fast path when the row fits in one block
    if BLOCK == N:
        offsets = tl.arange(0, BLOCK)
        x = tl.load(x_row_ptr + offsets, mask=None, eviction_policy='evict_first')
        x_sq = x * x
        sum_sq = tl.sum(x_sq, axis=0)
        mean_sq = sum_sq / N
        rstd = tl.rsqrt(mean_sq + eps)
        out = x * rstd
        tl.store(out_row_ptr + offsets, out, mask=None, eviction_policy='evict_last')
    else:
        # First pass: accumulate sum of squares
        acc = 0.0
        for col_start in range(0, N, BLOCK):
            offsets = col_start + tl.arange(0, BLOCK)
            mask = offsets < N
            x = tl.load(x_row_ptr + offsets, mask=mask, other=0.0, eviction_policy='evict_first')
            x_sq = x * x
            acc += tl.sum(x_sq, axis=0)
        mean_sq = acc / N
        rstd = tl.rsqrt(mean_sq + eps)

        # Second pass: rescale and store
        for col_start in range(0, N, BLOCK):
            offsets = col_start + tl.arange(0, BLOCK)
            mask = offsets < N
            x = tl.load(x_row_ptr + offsets, mask=mask, other=0.0, eviction_policy='evict_first')
            out = x * rstd
            tl.store(out_row_ptr + offsets, out, mask=mask, eviction_policy='evict_last')


def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous()
    M, N = x.shape
    out = torch.empty_like(x)

    # Tunable knobs – can be adjusted for performance tuning
    BLOCK = N          # one block per row (fast, single‑pass)
    num_warps = 8
    num_stages = 1

    grid = (M,)
    rms_norm_kernel[grid](
        x, out,
        N=N,
        eps=1e-5,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out