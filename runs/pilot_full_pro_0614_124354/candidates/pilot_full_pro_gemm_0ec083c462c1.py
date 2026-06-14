import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row and column ranges for the output tile
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]  # (BLOCK_M, 1)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]  # (1, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)                              # (BLOCK_K,)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offset = k * BLOCK_K
        rk_off = k_offset + rk

        # Load a tile of A[rm, rk_off]  (BLOCK_M, BLOCK_K)
        a_mask = (rm < M) & (rk_off[None, :] < K)
        a_ptr = A + rm * stride_am + rk_off[None, :] * stride_ak
        a = tl.load(a_ptr, mask=a_mask, other=0.0)

        # Load a tile of B[rk_off, rn]  (BLOCK_K, BLOCK_N)
        b_mask = (rk_off[:, None] < K) & (rn < N)
        b_ptr = B + rk_off[:, None] * stride_bk + rn * stride_bn
        b = tl.load(b_ptr, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    c = acc.to(tl.float16)
    c_ptr = C + rm * stride_cm + rn * stride_cn
    c_mask = (rm < M) & (rn < N)
    tl.store(c_ptr, c, mask=c_mask)


def triton_run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == (4096, 4096) and b.shape == (4096, 4096)
    assert a.dtype == torch.float16 and b.dtype == torch.float16

    M, K = a.shape
    N = b.shape[1]
    C = torch.empty((M, N), dtype=torch.float16, device=a.device)

    # Tuning knobs for follow‑up optimization
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )

    matmul_kernel[grid](
        a, b, C,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )
    return C