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
    # Program IDs
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Offsets inside a tile
    offs_m = tl.arange(0, BLOCK_M)  # (BLOCK_M,)
    offs_n = tl.arange(0, BLOCK_N)  # (BLOCK_N,)
    offs_k = tl.arange(0, BLOCK_K)  # (BLOCK_K,)

    # Base pointers for the start of the A and B tiles in the first k-step
    a_block_start = (
        A_ptr
        + (pid_m * BLOCK_M + offs_m[:, None]) * stride_am
        + offs_k[None, :] * stride_ak
    )
    b_block_start = (
        B_ptr
        + offs_k[:, None] * stride_bk
        + (pid_n * BLOCK_N + offs_n[None, :]) * stride_bn
    )

    # Accumulator in fp32 for precision
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    K_range = K // BLOCK_K  # exact due to divisibility
    for k in range(K_range):
        # Load with masks (always True, zero overhead)
        a = tl.load(
            a_block_start,
            mask=(pid_m * BLOCK_M + offs_m[:, None] < M)
            & (k * BLOCK_K + offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_block_start,
            mask=(k * BLOCK_K + offs_k[:, None] < K)
            & (pid_n * BLOCK_N + offs_n[None, :] < N),
            other=0.0,
        )

        acc += tl.dot(a, b)

        # Advance to next k-tile
        a_block_start += BLOCK_K * stride_ak
        b_block_start += BLOCK_K * stride_bk

    # Convert to fp16 and store
    c = acc.to(tl.float16)

    offs_c_m = pid_m * BLOCK_M + offs_m[:, None]
    offs_c_n = pid_n * BLOCK_N + offs_n[None, :]
    c_ptr = C_ptr + offs_c_m * stride_cm + offs_c_n * stride_cn

    tl.store(
        c_ptr,
        c,
        mask=(offs_c_m < M) & (offs_c_n < N),
    )


def triton_run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == (4096, 4096)
    assert b.shape == (4096, 4096)
    M, K_a = a.shape
    K_b, N = b.shape
    assert K_a == K_b
    K = K_a

    # Output allocation
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # Launch grid: one program per output tile
    grid = (triton.cdiv(M, 128), triton.cdiv(N, 128))

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
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=64,
        num_warps=8,
        num_stages=4,
    )
    return c