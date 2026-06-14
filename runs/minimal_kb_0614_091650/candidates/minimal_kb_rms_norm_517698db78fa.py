import torch
import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(
    x_ptr, out_ptr,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    # Grid-stride loop over rows assigned to this program
    base_row = tl.program_id(0) * ROWS_PER_PROGRAM
    for r in range(ROWS_PER_PROGRAM):
        row = base_row + r
        x_row_ptr = x_ptr + row * N
        out_row_ptr = out_ptr + row * N

        # Contiguous offsets, no mask needed since N == BLOCK
        offsets = tl.max_contiguous(tl.arange(0, BLOCK), BLOCK)
        x = tl.load(x_row_ptr + offsets, cache_modifier=".evict_first")
        # Compute RMS
        x_sq = x * x
        sum_sq = tl.sum(x_sq, axis=0)
        mean_sq = sum_sq / N
        rstd = tl.rsqrt(mean_sq + eps)
        out = x * rstd
        tl.store(out_row_ptr + offsets, out, cache_modifier=".evict_first")


def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous()
    M, N = x.shape
    out = torch.empty_like(x)

    # Tuned constants: process 4 rows per program to reduce launch overhead
    ROWS_PER_PROGRAM = 4
    BLOCK = N
    grid = (M // ROWS_PER_PROGRAM,)

    rms_norm_kernel[grid](
        x, out,
        N=N,
        eps=1e-5,
        BLOCK=BLOCK,
        ROWS_PER_PROGRAM=ROWS_PER_PROGRAM,
        num_warps=8,
        num_stages=1,
    )
    return out