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

    # Base pointers for this tile's first K-block
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # All dimensions are exactly divisible by the block sizes, no mask needed
    K_range = K // BLOCK_K
    for _ in range(K_range):
        a = tl.load(a_block_start)
        b = tl.load(b_block_start)
        acc += tl.dot(a, b)
        a_block_start += BLOCK_K * stride_ak
        b_block_start += BLOCK_K * stride_bk

    c = acc.to(tl.float16)

    # Output tile is fully inside C (no boundary mask required)
    offs_c_m = pid_m * BLOCK_M + offs_m[:, None]
    offs_c_n = pid_n * BLOCK_N + offs_n[None, :]
    c_ptr = C_ptr + offs_c_m * stride_cm + offs_c_n * stride_cn
    tl.store(c_ptr, c)


# Apply JIT without the decorator to keep the name clean
gemm_kernel = triton.jit(gemm_kernel_fn)


def triton_run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute a @ b for 4096x4096 float16 matrices on NVIDIA RTX 5090."""
    assert a.shape == (4096, 4096)
    assert b.shape == (4096, 4096)
    M, K_a = a.shape
    K_b, N = b.shape
    assert K_a == K_b
    K = K_a

    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # Tile sizes chosen for square 4096 matrices and tuned num_warps/num_stages
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    grid = (M // BLOCK_M, N // BLOCK_N)  # exact division

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
        num_warps=8,    # tuned balance of occupancy and register pressure
        num_stages=5,   # extra prefetch depth for memory-hungry pipelines
    )
    return c