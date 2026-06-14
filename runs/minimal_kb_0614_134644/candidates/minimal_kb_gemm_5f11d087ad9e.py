import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton GEMM kernel – specialized for A: f16[4096,4096], B: f16[4096,4096]
# Blocks are chosen so that M, N, K are exact multiples => no masks needed.
# Uses group_swizzle for improved L2 reuse.
# ---------------------------------------------------------------------------
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    # All dimensions are tl.constexpr (fixed at 4096)
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Swizzled program ID mapping (improves L2 hit rate)
    pid = tl.program_id(0)
    grid_n = N // BLOCK_N
    group_id = pid // grid_n
    group_size = tl.minimum(GROUP_M, M // BLOCK_M - group_id * GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % grid_n) // group_size * GROUP_M + (pid // group_size % GROUP_M)
    # note: corrected group_swizzle formula – each group covers GROUP_M blocks of M and GROUP_M blocks of N interleaved

    # Offsets for this tile (no masks because sizes are exact)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K-loop
    for start_k in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + (offs_m[:, None] * stride_am + (start_k + offs_k[None, :]) * stride_ak))
        b = tl.load(b_ptr + ((start_k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn))
        acc += tl.dot(a, b)

    c = acc.to(tl.float16)
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, c)


def triton_run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda
    assert a.dtype == torch.float16 and b.dtype == torch.float16
    M, K = a.shape
    _, N = b.shape
    # Check contiguity (row-major)
    assert a.stride(0) == K and a.stride(1) == 1
    assert b.stride(0) == N and b.stride(1) == 1

    c = torch.empty((M, N), device='cuda', dtype=torch.float16)

    # Tuned tile sizes for RTX 5090 (Blackwell, large L1)
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    GROUP_M = 8

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)  # flattened grid for swizzle

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        num_warps=8,
        num_stages=4,
    )
    return c