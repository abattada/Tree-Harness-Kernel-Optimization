import torch
import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(
    x_ptr, out_ptr,
    M: int,           # number of rows, 8192
    N: tl.constexpr,  # 4096
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    offsets = tl.arange(0, BLOCK)

    # Start with the row assigned to this program
    row = pid
    if row < M:
        # Load first row
        x0_ptr = x_ptr + row * N
        x0 = tl.load(x0_ptr + offsets, eviction_policy='evict_first')

        # Loop over remaining rows with double‑buffered prefetch
        next_row = row + num_programs
        while next_row < M:
            # Prefetch the next row while we compute the current one
            x1_ptr = x_ptr + next_row * N
            x1 = tl.load(x1_ptr + offsets, eviction_policy='evict_first')

            # Compute RMS for the current row (x0)
            x_sq = x0 * x0
            sum_sq = tl.sum(x_sq, axis=0)
            mean_sq = sum_sq / N
            rstd = tl.rsqrt(mean_sq + eps)
            out0 = x0 * rstd
            tl.store(out_ptr + row * N + offsets, out0,
                     eviction_policy='evict_last')

            # Advance to the next row
            row = next_row
            x0 = x1
            next_row += num_programs

        # Handle the final row (x0 still holds its data)
        x_sq = x0 * x0
        sum_sq = tl.sum(x_sq, axis=0)
        mean_sq = sum_sq / N
        rstd = tl.rsqrt(mean_sq + eps)
        out0 = x0 * rstd
        tl.store(out_ptr + row * N + offsets, out0,
                 eviction_policy='evict_last')


def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous(), "Input must be contiguous"
    M, N = x.shape          # 8192, 4096
    out = torch.empty_like(x)

    BLOCK = N               # one tile covers the whole row
    # Use fewer programs than rows to amortise launch overhead,
    # while keeping enough to saturate the GPU (4096 works well here).
    grid = (min(M // 2, 4096),)

    rms_norm_kernel[grid](
        x, out,
        M=M,
        N=N,
        eps=1e-5,
        BLOCK=BLOCK,
        num_warps=8,
        num_stages=1,       # single iteration, pipelining handled by software
    )
    return out