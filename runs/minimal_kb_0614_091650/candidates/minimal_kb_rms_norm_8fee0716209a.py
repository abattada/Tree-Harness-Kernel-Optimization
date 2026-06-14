import torch
import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(
    x_ptr, out_ptr,
    stride_x, stride_out,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    base_row = tl.program_id(0) * ROWS_PER_PROG
    cols = tl.arange(0, BLOCK_SIZE)

    for r in range(ROWS_PER_PROG):
        row = base_row + r
        x_row_ptr = x_ptr + row * stride_x
        out_row_ptr = out_ptr + row * stride_out

        x = tl.load(x_row_ptr + cols, mask=None)
        x_sq = x * x
        sum_sq = tl.sum(x_sq, axis=0)
        mean_sq = sum_sq / N
        rstd = tl.rsqrt(mean_sq + eps)
        out = x * rstd
        tl.store(out_row_ptr + cols, out, mask=None)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous()
    M, N = x.shape
    out = torch.empty_like(x)

    ROWS_PER_PROG = 4   # process 4 rows per program
    grid = (M // ROWS_PER_PROG,)

    rms_norm_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        N=N,
        eps=1e-5,
        BLOCK_SIZE=N,          # N is power of 2, exactly matches BLOCK_SIZE
        ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=8,
        num_stages=1,
    )
    return out