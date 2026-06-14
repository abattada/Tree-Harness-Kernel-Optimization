import torch
import triton
import triton.language as tl


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

    # Build base pointer blocks for a (columns [0..N)) and b (columns [N..2N))
    a_base = x_ptr + rows_2d * stride_x0 + cols_2d
    b_base = x_ptr + rows_2d * stride_x0 + (cols_2d + N)
    out_base = out_ptr + rows_2d * stride_out0 + cols_2d

    # Hint 128‑byte alignment for wide, coalesced, vectorized loads/stores.
    # Both M and N are exact multiples of the tiling, so no mask is needed.
    a_ptrs = tl.multiple_of(a_base, (128,))
    b_ptrs = tl.multiple_of(b_base, (128,))
    out_ptrs = tl.multiple_of(out_base, (128,))

    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)

    silu_a = a * tl.sigmoid(a)       # SiLU(x) = x * σ(x)
    out_val = silu_a * b

    tl.store(out_ptrs, out_val)


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

    # Tiling that covers the exact output shape (no boundary masks needed)
    BLOCK_M, BLOCK_N = 8, 256
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    swiglu_kernel[grid](
        x,
        out,
        M,
        N,
        x.stride(0),
        out.stride(0),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )
    return out