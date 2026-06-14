import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton GEMM kernel for fp16 matrix multiplication (A @ B)
# Specialized for M = N = K = 4096, which are exact multiples of block sizes.
# Uses group-swizzled program ordering to improve L2 cache reuse.
# ---------------------------------------------------------------------------
@triton.jit
def gemm_kernel(
    A, B, C,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    M: tl.constexpr = 4096,
    N: tl.constexpr = 4096,
    K: tl.constexpr = 4096,
):
    # Program ID in a 1D grid
    pid = tl.program_id(0)

    # Number of blocks along M and N (exact, no remainder)
    num_pid_m = M // BLOCK_M
    num_pid_n = N // BLOCK_N

    # Swizzle: groups of GROUP_M consecutive M-blocks share an N-block
    group_id = pid // (num_pid_n * GROUP_M)
    first_m = group_id * GROUP_M
    group_size_m = min(GROUP_M, num_pid_m - first_m)

    pid_in_group = pid % (num_pid_n * group_size_m)
    n = pid_in_group // group_size_m
    m = first_m + (pid_in_group % group_size_m)

    # Guard against out-of-bounds (should never trigger with exact division)
    if m >= num_pid_m or n >= num_pid_n:
        return

    # Block offsets
    offs_m = m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to first tile of A and B (k=0)
    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # fp32 accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K-loop with streaming loads to avoid cache pollution
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, cache_modifier=".evict_first")
        b = tl.load(b_ptrs, cache_modifier=".evict_first")
        acc += tl.dot(a, b, input_precision="ieee")
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Store result as fp16 with streaming hint
    offs_cm = m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    tl.store(c_ptrs, acc.to(tl.float16), cache_modifier=".streaming")


# ---------------------------------------------------------------------------
# Public wrapper: allocates output and launches the kernel
# ---------------------------------------------------------------------------
def triton_run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Input validation (fixed shape and dtype)
    assert a.dtype == torch.float16 and b.dtype == torch.float16
    assert a.shape == (4096, 4096) and b.shape == (4096, 4096)
    M, K = a.shape
    _, N = b.shape
    assert K == 4096 and N == 4096

    # Output buffer
    c = torch.empty((M, N), dtype=torch.float16, device=a.device)

    # Tile and launch parameters (tuned for RTX 5090)
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    GROUP_M = 8
    num_warps = 8
    num_stages = 4

    # Grid: one program per output tile (exact number due to perfect division)
    grid_m = M // BLOCK_M   # 32
    grid_n = N // BLOCK_N   # 32
    grid = (grid_m * grid_n,)  # 1024 programs

    # Launch
    gemm_kernel[grid](
        a, b, c,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
        GROUP_M,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return c