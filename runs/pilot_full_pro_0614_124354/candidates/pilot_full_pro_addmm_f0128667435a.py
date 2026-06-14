import torch
import triton
import triton.language as tl


@triton.jit
def _fused_bias_matmul_kernel(
    bias_ptr, a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_tile_ptr = a_ptr + offs_m[:, None] * stride_am
    b_tile_ptr = b_ptr + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_tile_ptr + (k + offs_k[None, :]) * stride_ak,
                    eviction_policy='evict_first')
        b = tl.load(b_tile_ptr + (k + offs_k[:, None]) * stride_bk,
                    eviction_policy='evict_first')
        acc += tl.dot(a, b)

    bias = tl.load(bias_ptr + offs_m)
    acc += bias[:, None]

    c = acc.to(tl.float16)
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, c)


def triton_run(bias: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    assert bias.shape == (M,), "bias must be 1D [M]"
    assert a.shape == (M, K) and b.shape == (K, N), "dimension mismatch"

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _fused_bias_matmul_kernel[grid](
        bias, a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=8,
        num_stages=5,
    )
    return c