import torch
import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(
    x_ptr, out_ptr,
    N: tl.constexpr,
    eps: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROG
    offsets = tl.arange(0, BLOCK)

    # Base pointers for the first row of this program; add alignment hint.
    # N is a multiple of 16, so each row start is at least 16-element aligned.
    x_base = tl.multiple_of(x_ptr + row_start * N, 16)
    out_base = tl.multiple_of(out_ptr + row_start * N, 16)

    for i in range(ROWS_PER_PROG):
        row_off = i * N
        x_row = tl.multiple_of(x_base + row_off, 16)
        out_row = tl.multiple_of(out_base + row_off, 16)

        # Coalesced load of one full row; input is streaming data.
        x = tl.load(x_row + offsets, mask=None, eviction_policy='evict_first')

        # Single-pass RMS normalisation.
        x_sq = x * x
        sum_sq = tl.sum(x_sq, axis=0)
        mean_sq = sum_sq / N
        rstd = tl.rsqrt(mean_sq + eps)

        out = x * rstd
        # Store the result; output is likely to be reused soon.
        tl.store(out_row + offsets, out, mask=None, eviction_policy='evict_last')


def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous()
    M, N = x.shape  # 8192, 4096
    out = torch.empty_like(x)

    ROWS_PER_PROG = 2
    BLOCK = N          # One tile covers the whole row
    grid = (M // ROWS_PER_PROG,)

    rms_norm_kernel[grid](
        x, out,
        N=N,
        eps=1e-5,
        ROWS_PER_PROG=ROWS_PER_PROG,
        BLOCK=BLOCK,
        num_warps=16,
        num_stages=2,
    )
    return out