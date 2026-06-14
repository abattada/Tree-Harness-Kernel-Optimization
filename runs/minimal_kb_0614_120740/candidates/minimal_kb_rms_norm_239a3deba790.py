import torch
import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(
    x_ptr, out_ptr,
    M, N,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROG
    for i in tl.static_range(ROWS_PER_PROG):
        row = row_start + i
        if row < M:
            x_row_ptr = x_ptr + row * N
            out_row_ptr = out_ptr + row * N
            offsets = tl.arange(0, BLOCK)

            # Load input row – streaming, hint evict first
            x = tl.load(x_row_ptr + offsets, mask=None, eviction_policy='evict_first')

            # Compute RMS
            x_sq = x * x
            sum_sq = tl.sum(x_sq, axis=0)
            mean_sq = sum_sq / N
            rstd = tl.rsqrt(mean_sq + eps)

            out = x * rstd
            # Store output – may be reused later, hint evict last
            tl.store(out_row_ptr + offsets, out, mask=None, eviction_policy='evict_last')


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    x: (8192, 4096) float32, contiguous.
    Returns: same shape, rms-normalized.
    """
    assert x.is_contiguous()
    M, N = x.shape
    out = torch.empty_like(x)

    BLOCK = N          # 4096, exactly one block per row
    ROWS_PER_PROG = 2  # process two rows per program to reduce launch overhead

    grid = (triton.cdiv(M, ROWS_PER_PROG),)
    rms_norm_kernel[grid](
        x, out,
        M, N,
        eps=1e-5,
        BLOCK=BLOCK,
        ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=8,
        num_stages=1,
    )
    return out