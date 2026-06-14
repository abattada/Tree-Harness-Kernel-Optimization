import torch
import triton
import triton.language as tl


@triton.jit
def addmm_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Static compile-time safety: all dims are multiples of block sizes
    tl.static_assert(M % BLOCK_M == 0)
    tl.static_assert(N % BLOCK_N == 0)
    tl.static_assert(K % BLOCK_K == 0)

    pid_m = tl.program_id(1)  # row index
    pid_n = tl.program_id(0)  # column index

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    # Offsets for the tile
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to A and B tiles (bias pointer is 1D)
    a_tile_ptr = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_tile_ptr = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main K loop
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_tile_ptr + k * stride_ak)
        b = tl.load(b_tile_ptr + k * stride_bk)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    # Load bias tile and broadcast across rows
    bias_tile = tl.load(bias_ptr + offs_n)  # shape (BLOCK_N,)
    acc += bias_tile[None, :]

    # Store output
    out_tile = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_tile, acc.to(tl.float16))


def triton_run(bias: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute out = bias + a @ b, where bias is a vector of length N.
    All inputs are fp16; shapes are [4096], [4096, 4096], [4096, 4096].
    Returns fp16 tensor of shape [4096, 4096].
    """
    assert bias.dtype == torch.float16 and a.dtype == torch.float16 and b.dtype == torch.float16
    M, K = a.shape
    N = b.shape[1]

    out = torch.empty((M, N), dtype=torch.float16, device=a.device)

    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 32

    grid = (N // BLOCK_N, M // BLOCK_M)  # (x, y)
    addmm_kernel[grid](
        a,
        b,
        bias,
        out,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )
    return out