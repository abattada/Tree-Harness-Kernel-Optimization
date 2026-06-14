import torch
import triton
import triton.language as tl


@triton.jit
def gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,  # for a: row stride (M dimension) and column stride (K dimension)
    stride_bk, stride_bn,  # for b: row stride (K dim) and column stride (N dim)
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Tiled FP16 matrix multiplication: C = A @ B."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile extent offsets (no boundary mask needed because 4096 is divisible by the chosen block sizes)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # FP32 accumulator for the tile
    c_tile = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over the K dimension in tiles
    for k in range(0, K, BLOCK_K):
        # Load a tile from A: shape (BLOCK_M, BLOCK_K)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak
        a_tile = tl.load(a_ptrs)

        # Load a tile from B: shape (BLOCK_K, BLOCK_N)
        b_ptrs = b_ptr + (k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_tile = tl.load(b_ptrs)

        # Accumulate using tensor cores (fp16 inputs, fp32 accumulate)
        c_tile += tl.dot(a_tile, b_tile)

    # Write the result tile back to global memory, converting to FP16
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c_tile.to(tl.float16))


def triton_run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K_a = a.shape
    K_b, N = b.shape
    assert K_a == K_b, "Inner dimensions must match"
    K = K_a
    c = torch.empty((M, N), dtype=torch.float16, device=a.device)

    # Tuning knobs (can be swept later)
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    gemm_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )
    return c