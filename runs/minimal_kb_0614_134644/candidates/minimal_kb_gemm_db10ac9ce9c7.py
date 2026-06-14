import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton GEMM kernel: C = A * B (matrix multiplication, all fp16)
# ---------------------------------------------------------------------------
@triton.jit
def gemm_kernel_swizzled(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Program ID (1D grid)
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # Group swizzle: consecutive M-blocks share the same N-block to improve L2 reuse
    group_id = pid // (num_pid_n * GROUP_M)
    first_m = group_id * GROUP_M
    group_size_m = min(GROUP_M, num_pid_m - first_m)
    pid_in_group = pid % (num_pid_n * group_size_m)
    n = pid_in_group // group_size_m
    m = first_m + (pid_in_group % group_size_m)

    if m >= num_pid_m or n >= num_pid_n:
        return

    # Block tile ranges
    offs_am = m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Accumulator in fp32 for precision
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K loop with streaming loads
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, cache_modifier=".evict_first")
        b = tl.load(b_ptrs, cache_modifier=".evict_first")
        acc += tl.dot(a, b, input_precision="ieee")
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Store result in fp16
    offs_cm = m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    tl.store(c_ptrs, acc.to(tl.float16), cache_modifier=".streaming")


# ---------------------------------------------------------------------------
# Wrapper that allocates output and launches the kernel
# ---------------------------------------------------------------------------
def triton_run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Inputs are fp16, shape (4096, 4096) each
    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb

    c = torch.empty((M, N), dtype=torch.float16, device=a.device)

    # Tuned block sizes for RTX 5090 (Blackwell) – 128x128 tiles, K=32
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    GROUP_M = 8
    num_warps = 8
    num_stages = 4

    grid_m = triton.cdiv(M, BLOCK_M)
    grid_n = triton.cdiv(N, BLOCK_N)
    grid = (grid_m * grid_n,)  # 1D grid with swizzle

    gemm_kernel_swizzled[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
        GROUP_M,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return c