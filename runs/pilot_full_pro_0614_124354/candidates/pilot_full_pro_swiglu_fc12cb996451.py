import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # candidate BLOCK_M x BLOCK_N with default num_warps=4
        triton.Config({'BLOCK_M': 8,  'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 8,  'BLOCK_N': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256}, num_warps=4, num_stages=2),
        # higher occupancy variants
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128,'BLOCK_N': 128}, num_warps=4, num_stages=2),
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

    a_ptrs = x_ptr + rows_2d * stride_x0 + cols_2d
    b_ptrs = x_ptr + rows_2d * stride_x0 + (cols_2d + N)
    out_ptrs = out_ptr + rows_2d * stride_out0 + cols_2d

    a = tl.load(a_ptrs, mask=mask, other=0.0)
    b = tl.load(b_ptrs, mask=mask, other=0.0)

    silu_a = a * tl.sigmoid(a)
    out_val = silu_a * b

    tl.store(out_ptrs, out_val, mask=mask)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2 and x.shape[1] % 2 == 0, "Input must be 2D with even last dim"
    M, D = x.shape
    N = D // 2
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # grid depends on the chosen block sizes; lamdba captures M, N
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

    swiglu_kernel[grid](
        x,
        out,
        M,
        N,
        x.stride(0),
        out.stride(0),
    )
    return out