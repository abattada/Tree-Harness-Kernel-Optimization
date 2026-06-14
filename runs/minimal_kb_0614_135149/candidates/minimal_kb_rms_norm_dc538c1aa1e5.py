import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    x_ptr,
    out_ptr,
    M,                     # number of rows (dynamic, but known at launch)
    N: tl.constexpr,       # row length (constant for specialization)
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    """
    Each program processes ROWS_PER_PROG contiguous rows.
    Reduces launch overhead while keeping the same single‑row RMS logic.
    N must equal BLOCK (i.e., one tile covers an entire row).
    """
    pid = tl.program_id(0)
    base_row = pid * ROWS_PER_PROG

    offsets = tl.arange(0, BLOCK)

    # Process up to ROWS_PER_PROG rows (guard against partial tail)
    for r in range(ROWS_PER_PROG):
        row = base_row + r
        if row < M:
            x_row_ptr = x_ptr + row * N
            out_row_ptr = out_ptr + row * N

            # Load entire row with streaming hint
            x = tl.load(x_row_ptr + offsets, mask=None, eviction_policy="evict_first")

            # Single‑pass RMS normalization
            x_sq = x * x
            sum_sq = tl.sum(x_sq, axis=0)
            mean_sq = sum_sq / N
            rstd = tl.rsqrt(mean_sq + eps)

            out = x * rstd

            # Store result – hint to keep in cache if reused later
            tl.store(out_row_ptr + offsets, out, mask=None, eviction_policy="evict_last")


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    RMS normalization: out = x * rsqrt(mean(x^2, dim=-1) + 1e-5).
    Input (float32, shape [M, N]) must be contiguous.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    M, N = x.shape

    out = torch.empty_like(x)

    BLOCK = N                     # one tile per row
    ROWS_PER_PROG = 2             # each block processes 2 rows (reduces grid size)
    grid = (M + ROWS_PER_PROG - 1) // ROWS_PER_PROG   # ceil division

    rms_norm_kernel[grid](
        x, out,
        M, N,
        eps=1e-5,
        BLOCK=BLOCK,
        ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=8,              # good balance for 4096‑element rows
        num_stages=1,
    )

    return out