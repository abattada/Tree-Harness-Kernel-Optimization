import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128}, num_warps=w, num_stages=s)
        for w in [2, 4, 8, 16]
        for s in [2, 3, 4, 5, 6]
    ],
    key=['M', 'N'],
)
@triton.jit
def swiglu_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_x0,
    stride_out0,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    rows_2d = rows[:, None]
    cols_2d = cols[None, :]

    mask = (rows_2d < M) & (cols_2d < N)

    # base pointers for a and b: a in [0..N), b in [N..2N)
    a_ptrs = x_ptr + rows_2d * stride_x0 + cols_2d
    b_ptrs = x_ptr + rows_2d * stride_x0 + (cols_2d + N)
    out_ptrs = out_ptr + rows_2d * stride_out0 + cols_2d

    a = tl.load(a_ptrs, mask=mask, other=0.0)
    b = tl.load(b_ptrs, mask=mask, other=0.0)

    silu_a = a * tl.sigmoid(a)          # SiLU(x) = x * σ(x)
    out_val = silu_a * b

    tl.store(out_ptrs, out_val, mask=mask)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x:  float32 tensor of shape [8192, 8192]
    Returns:
        out: float32 tensor of shape [8192, 4096]
            out = silu(x[..., :4096]) * x[..., 4096:]
    """
    assert x.ndim == 2 and x.shape[1] % 2 == 0, "Input must be 2D with even last dim"
    M, D = x.shape
    N = D // 2
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )

    swiglu_kernel[grid](
        x,
        out,
        M,
        N,
        x.stride(0),
        out.stride(0),
    )
    return out