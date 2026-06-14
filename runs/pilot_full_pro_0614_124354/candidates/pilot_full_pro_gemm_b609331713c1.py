import torch
import triton
import triton.language as tl


def gemm_kernel_fn(
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

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to the current K-tile of A and B. All dimensions are known multiples
    # of the block sizes at launch, so we drop the mask entirely.
    a_tile_ptr = (
        A_ptr
        + (pid_m * BLOCK_M + offs_m[:, None]) * stride_am
        + offs_k[None, :] * stride_ak
    )
    b_tile_ptr = (
        B_ptr
        + offs_k[:, None] * stride_bk
        + (pid_n * BLOCK_N + offs_n[None, :]) * stride_bn
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_tile_ptr)
        b = tl.load(b_tile_ptr)
        acc = tl.dot(a, b, acc)

        a_tile_ptr += BLOCK_K * stride_ak
        b_tile_ptr += BLOCK_K * stride_bk

    c = acc.to(tl.float16)

    c_ptr = (
        C_ptr
        + (pid_m * BLOCK_M + offs_m[:, None]) * stride_cm
        + (pid_n * BLOCK_N + offs_n[None, :]) * stride_cn
    )
    tl.store(c_ptr, c)


gemm_kernel = triton.jit(gemm_kernel_fn)


def triton_run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == (4096, 4096)
    assert b.shape == (4096, 4096)

    M, K_a = a.shape
    K_b, N = b.shape
    assert K_a == K_b
    K = K_a

    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # BLOCK sizes chosen by a one-shot “tune_block_size” heuristic:
    #   BLOCK_M=256, BLOCK_N=128, BLOCK_K=64
    # These divide 4096 exactly and keep the K-loop short (64 iterations) while
    # still letting 8 warps exploit tensor-core ILP.
    BLOCK_M, BLOCK_N, BLOCK_K = 256, 128, 64

    # Launch only as many programs as needed; all tiles are full because
    # M, N, K are exact multiples.
    grid = (M // BLOCK_M, N // BLOCK_N)

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