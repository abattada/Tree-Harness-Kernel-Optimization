import torch
import triton
import triton.language as tl


@triton.jit
def gemm_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = tl.arange(0, BLOCK_M)[:, None]  # (BLOCK_M, 1)
    offs_n = tl.arange(0, BLOCK_N)[None, :]  # (1, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)           # (BLOCK_K,)

    # Base pointers for A and B tiles
    a_base = (
        A_ptr
        + (pid_m * BLOCK_M + offs_m) * stride_am
        + offs_k[None, :] * stride_ak
    )
    b_base = (
        B_ptr
        + offs_k[:, None] * stride_bk
        + (pid_n * BLOCK_N + offs_n) * stride_bn
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over the K dimension in steps of BLOCK_K
    for k in range(0, K, BLOCK_K):
        # Load A tile with mask along M and K
        a_tile = tl.load(
            a_base,
            mask=((pid_m * BLOCK_M + offs_m) < M) & ((k + offs_k[None, :]) < K),
            other=0.0,
        )
        # Load B tile with mask along K and N
        b_tile = tl.load(
            b_base,
            mask=((k + offs_k[:, None]) < K) & ((pid_n * BLOCK_N + offs_n) < N),
            other=0.0,
        )

        # Accumulate with tensor-core operations
        acc = tl.dot(a_tile, b_tile, acc)

        # Advance to the next K block
        a_base += BLOCK_K * stride_ak
        b_base += BLOCK_K * stride_bk

    # Convert acc to fp16 and store
    c = acc.to(tl.float16)

    offs_c_m = pid_m * BLOCK_M + offs_m
    offs_c_n = pid_n * BLOCK_N + offs_n
    c_ptr = C_ptr + offs_c_m * stride_cm + offs_c_n * stride_cn

    tl.store(c_ptr, c, mask=(offs_c_m < M) & (offs_c_n < N))


def triton_run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == (4096, 4096), f"Expected a shape (4096,4096) got {a.shape}"
    assert b.shape == (4096, 4096), f"Expected b shape (4096,4096) got {b.shape}"
    M, K = a.shape
    N = b.shape[1]
    assert K == b.shape[0], "Inner dimensions must match"

    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # Tuned for RTX 5090 Blackwell: larger M tile to improve occupancy vs register pressure
    BLOCK_M = 256
    BLOCK_N = 128
    BLOCK_K = 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    gemm_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=8,
        num_stages=4,
    )
    return c