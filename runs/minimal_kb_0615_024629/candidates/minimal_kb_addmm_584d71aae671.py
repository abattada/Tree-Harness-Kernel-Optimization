import torch
import triton
import triton.language as tl

@triton.jit
def addmm_kernel(
    a_ptr, b_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_outm, stride_outn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m[:, None] < M
    n_mask = offs_n[None, :] < N

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    k_end = tl.cdiv(K, BLOCK_K) * BLOCK_K
    for k in range(0, k_end, BLOCK_K):
        a_tile = tl.load(
            a_ptr + offs_m[:, None] * stride_am + (offs_k + k)[None, :] * stride_ak,
            mask=m_mask & ((offs_k + k)[None, :] < K),
            other=0.0,
        )
        b_tile = tl.load(
            b_ptr + (offs_k + k)[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=((offs_k + k)[:, None] < K) & n_mask,
            other=0.0,
        )
        acc = tl.dot(a_tile, b_tile, acc)

    bias_tile = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias_tile[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn
    tl.store(out_ptrs, acc.to(tl.float16), mask=m_mask & n_mask)


def triton_run(bias: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_contiguous() and b.is_contiguous(), "inputs must be contiguous"
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    assert bias.shape == (N,)

    out = torch.empty((M, N), dtype=a.dtype, device=a.device)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    addmm_kernel[grid](
        a, b, bias, out,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=8,
        num_stages=4,
    )

    return out