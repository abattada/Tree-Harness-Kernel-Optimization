import torch
import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(
    x_ptr, out_ptr,
    stride_x, stride_out,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
    BLOCK_COL: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_ROW
    rows = row_start + tl.arange(0, BLOCK_ROW)
    cols = tl.arange(0, BLOCK_COL)

    # Load a 2D tile of shape (BLOCK_ROW, BLOCK_COL) – no mask because N == BLOCK_COL
    x = tl.load(
        x_ptr + rows[:, None] * stride_x + cols[None, :],
        eviction_policy='evict_first',
    )
    x_sq = x * x
    sum_sq = tl.sum(x_sq, axis=1)          # (BLOCK_ROW,)
    mean_sq = sum_sq / N
    rstd = tl.rsqrt(mean_sq + eps)         # (BLOCK_ROW,)
    out = x * rstd[:, None]                # broadcast
    tl.store(
        out_ptr + rows[:, None] * stride_out + cols[None, :],
        out,
        eviction_policy='evict_first',
    )

def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous()
    M, N = x.shape
    out = torch.empty_like(x)

    BLOCK_ROW = 4          # process 4 rows per program – good balance
    BLOCK_COL = N          # 4096, equals N so no masking needed
    grid = (M // BLOCK_ROW,)   # 2048 programs

    rms_norm_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        N=N,
        eps=1e-5,
        BLOCK_ROW=BLOCK_ROW,
        BLOCK_COL=BLOCK_COL,
        num_warps=8,
        num_stages=1,
    )
    return out