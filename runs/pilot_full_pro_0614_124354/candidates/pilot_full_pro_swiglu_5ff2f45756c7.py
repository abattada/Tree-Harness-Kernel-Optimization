import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_kernel(
    x_ptr,          # input: [M, 2*N_OUT]
    y_ptr,          # output: [M, N_OUT]
    M: int,
    N_OUT: int,
    stride_x_m: int,
    stride_x_n: int,
    stride_y_m: int,
    stride_y_n: int,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Row and column offsets for this tile
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # shape (BLOCK_M,)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # shape (BLOCK_N,)

    # Masks (unnecessary for exact shape, but kept for generality)
    mask_m = rm[:, None] < M               # (BLOCK_M, 1)
    mask_n = rn[None, :] < N_OUT           # (1, BLOCK_N)

    # Base pointers for each row in the tile
    x_base = x_ptr + rm[:, None] * stride_x_m   # (BLOCK_M, 1)
    y_base = y_ptr + rm[:, None] * stride_y_m   # (BLOCK_M, 1)

    # Pointers for 'a' (first half) and 'b' (second half)
    a_ptrs = x_base + rn[None, :] * stride_x_n          # columns [0, N_OUT)
    b_ptrs = x_base + (rn[None, :] + N_OUT) * stride_x_n # columns [N_OUT, 2*N_OUT)

    a = tl.load(a_ptrs, mask=mask_m & mask_n, other=0.0)
    b = tl.load(b_ptrs, mask=mask_m & mask_n, other=0.0)

    # SiLU: a * sigmoid(a)
    silu_a = a * tl.sigmoid(a)
    y = silu_a * b

    y_ptrs = y_base + rn[None, :] * stride_y_n
    tl.store(y_ptrs, y, mask=mask_m & mask_n)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    x: shape [M, 2*N_OUT], float32
    returns: shape [M, N_OUT], float32
    """
    assert x.ndim == 2 and x.shape[-1] % 2 == 0
    M, N2 = x.shape
    N_OUT = N2 // 2
    y = torch.empty((M, N_OUT), device=x.device, dtype=x.dtype)

    # Strides (contiguous layout)
    stride_x_n = 1
    stride_x_m = N2
    stride_y_n = 1
    stride_y_m = N_OUT

    # Tiling parameters – chosen for wide vectorization and good occupancy
    BLOCK_M = 4
    BLOCK_N = 512

    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N_OUT, BLOCK_N),
    )

    _swiglu_kernel[grid](
        x, y,
        M, N_OUT,
        stride_x_m, stride_x_n,
        stride_y_m, stride_y_n,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return y