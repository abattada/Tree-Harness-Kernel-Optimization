import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    x_ptr,      # input f32[8192,4096]
    out_ptr,    # output f32[8192,4096]
    M,          # number of rows (not constexpr to allow grid-stride loop)
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Each program processes multiple rows in a grid-stride loop.
    This reduces launch overhead and allows better memory latency hiding.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    # Iterate over rows assigned to this program
    for row in range(pid, M, num_programs):
        x_row_ptr = x_ptr + row * N
        out_row_ptr = out_ptr + row * N

        offsets = tl.arange(0, BLOCK)

        # Load entire row – data is contiguous, no mask needed
        x = tl.load(x_row_ptr + offsets, mask=None, eviction_policy="evict_first")

        # Single-pass RMS normalization
        x_sq = x * x
        sum_sq = tl.sum(x_sq, axis=0)          # sum over the row
        mean_sq = sum_sq / N
        rstd = tl.rsqrt(mean_sq + eps)

        out = x * rstd

        # Store result – hint that output may be reused later
        tl.store(out_row_ptr + offsets, out, mask=None, eviction_policy="evict_last")


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    RMS normalization along the last dimension (f32[8192,4096] -> f32[8192,4096]).
    x must be contiguous.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    M, N = x.shape

    out = torch.empty_like(x)

    # One tile covers the full row – no mask needed
    BLOCK = N                # 4096
    # Launch a configurable number of thread blocks; each will stride over rows.
    # For 8192 rows, 2048 blocks give 4 rows per program – a good balance.
    num_programs = max(1, min(2048, M))

    grid = (num_programs,)

    rms_norm_kernel[grid](
        x, out,
        M=M,
        N=N,
        eps=1e-5,
        BLOCK=BLOCK,
        num_warps=8,   # 256 threads → each thread processes 16 elements
        num_stages=1,  # no pipeline needed for a single-pass kernel
    )

    return out