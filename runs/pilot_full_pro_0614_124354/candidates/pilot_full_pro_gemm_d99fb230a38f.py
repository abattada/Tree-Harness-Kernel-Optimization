import torch
import triton
import triton.language as tl

# Problem dimensions are fixed, so we specialize them.
M, N, K = 4096, 4096, 4096

# Tuned tile sizes for RTX 5090 (sm_120) with fp16 tensor cores.
BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 64


@triton.jit
def gemm_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program IDs in a 2D grid of output tiles
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Tile‑internal offsets
    offs_m = tl.arange(0, BLOCK_M)          # shape (BLOCK_M, 1) via broadcasting
    offs_n = tl.arange(0, BLOCK_N)          # shape (1, BLOCK_N) via broadcasting
    offs_k = tl.arange(0, BLOCK_K)

    # Starting pointers for the first K‑tile
    a_block_ptr = (
        A_ptr
        + (pid_m * BLOCK_M + offs_m[:, None]) * stride_am
        + offs_k[None, :] * stride_ak
    )
    b_block_ptr = (
        B_ptr
        + offs_k[:, None] * stride_bk
        + (pid_n * BLOCK_N + offs_n[None, :]) * stride_bn
    )

    # Accumulator in fp32 for precision
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K‑loop; all accesses are in‑bounds because M, N, K divide the tile sizes evenly.
    for _ in range(K // BLOCK_K):
        a = tl.load(a_block_ptr)
        b = tl.load(b_block_ptr)
        acc += tl.dot(a, b)

        a_block_ptr += BLOCK_K * stride_ak
        b_block_ptr += BLOCK_K * stride_bk

    # Convert to fp16 and store the tile (no mask needed, grid is exact)
    c = acc.to(tl.float16)

    offs_c_m = pid_m * BLOCK_M + offs_m[:, None]
    offs_c_n = pid_n * BLOCK_N + offs_n[None, :]
    c_ptr = C_ptr + offs_c_m * stride_cm + offs_c_n * stride_cn
    tl.store(c_ptr, c)


def triton_run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == (M, K)
    assert b.shape == (K, N)
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # One program per output tile; the grid is exactly covered.
    grid = (M // BLOCK_M, N // BLOCK_N)

    gemm_kernel[grid](
        a,
        b,
        c,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        M=M,
        N=N,
        K=K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=16,
        num_stages=4,
    )
    return c