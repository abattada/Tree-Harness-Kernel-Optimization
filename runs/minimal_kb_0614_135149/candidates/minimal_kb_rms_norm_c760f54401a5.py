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
    ROWS_PER_PROG: tl.constexpr,
    M: tl.constexpr,
):
    """
    Persistent kernel with static loop over ROWS_PER_PROG rows.
    Pipelining (num_stages=2) overlaps loads of the next row with
    computation of the current row, hiding memory latency.
    """
    tl.static_assert(M % ROWS_PER_PROG == 0, "M must be divisible by ROWS_PER_PROG")
    tl.static_assert(N % BLOCK == 0, "N must be divisible by BLOCK")

    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROG
    offsets = tl.arange(0, BLOCK)

    # Static loop – Triton can pipeline iterations.
    for i in tl.static_range(ROWS_PER_PROG):
        row = row_start + i
        x_row_ptr = x_ptr + row * N
        out_row_ptr = out_ptr + row * N

        # Load entire row with streaming eviction hint.
        x = tl.load(x_row_ptr + offsets, mask=None, eviction_policy="evict_first")
        x_sq = x * x
        sum_sq = tl.sum(x_sq, axis=0)
        mean_sq = sum_sq / N
        rstd = tl.rsqrt(mean_sq + eps)
        out = x * rstd

        # Store result – evict_last may help if downstream kernels reuse it.
        tl.store(out_row_ptr + offsets, out, mask=None, eviction_policy="evict_last")


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    RMS normalization: x * rsqrt(mean(x^2, dim=-1) + 1e-5).
    Input: f32[8192, 4096]  (M=8192, N=4096)
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    M, N = x.shape

    # Constraints for the optimized kernel path.
    assert N == 4096 and M == 8192, "This kernel is specialized for [8192, 4096]"

    BLOCK = N                     # one full row per tile
    ROWS_PER_PROG: tl.constexpr = 8  # each program processes 8 rows
    grid = (M // ROWS_PER_PROG,)

    out = torch.empty_like(x)

    rms_norm_kernel[grid](
        x,
        out,
        N=N,
        eps=1e-5,
        BLOCK=BLOCK,
        ROWS_PER_PROG=ROWS_PER_PROG,
        M=M,
        num_warps=8,       # good occupancy – register pressure well within limits
        num_stages=2,      # pipelining hides memory latency
    )

    return out