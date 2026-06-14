import torch
import triton
import triton.language as tl
import math

# ---------------------------------------------------------------------------
# Triton kernel for GEMM: C = A @ B  (all fp16)
# ---------------------------------------------------------------------------
@triton.jit
def gemm_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Shape parameters (constexpr for this fixed size, but kept general via arguments)
    M, N, K,
    # Block sizes (constexpr)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # -----------------------------------------------------------------------
    # Program ID and grouped ordering
    # -----------------------------------------------------------------------
    pid_m = tl.program_id(0)  # along M dimension
    pid_n = tl.program_id(1)  # along N dimension
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # Swizzle: group of GROUP_M consecutive M blocks map to contiguous N blocks
    group_id = pid_m // GROUP_M
    first_n_pid = group_id * GROUP_M
    group_size_m = min(GROUP_M, num_pid_m - group_id * GROUP_M)
    pid_m_in_group = pid_m % GROUP_M
    pid_m = first_n_pid + pid_m_in_group
    pid_n = pid_n * GROUP_M + (pid_m_in_group % GROUP_M)  # not used directly – we use swizzle ordering
    # Actually simpler: use a linearized ordinal and compute swizzled (m, n)
    # Let's do standard triton swizzle.
    # We'll use pid as linear index and convert.
    pass  # We'll replace with simpler approach below

# Simplification: we will use a 1D grid of programs and compute (m, n) with swizzle.
# Let's rewrite the kernel with a single program ID and group-swizzled mapping.
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
    # Program ID
    pid = tl.program_id(0)
    # Number of programs along M and N dimensions
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    # Group ordering: consecutive M blocks share the same N block index
    # Group size = GROUP_M
    # In each group, we have GROUP_M * num_pid_m? Actually we interleave:
    # For each group of GROUP_M M-blocks, process a N-block.
    # So grid is (num_pid_m * num_pid_n,)
    group_id = pid // (num_pid_n * GROUP_M)              # group index along M
    first_m = group_id * GROUP_M                          # first m block in this group
    group_size_m = min(GROUP_M, num_pid_m - first_m)     # number of m blocks in this group
    # Within the group, assign pids to (m, n) in column-major order
    # (n varies fastest within a group)
    pid_in_group = pid % (num_pid_n * group_size_m)
    n = pid_in_group // group_size_m
    m = first_m + (pid_in_group % group_size_m)
    # Now m and n are the block indices
    if m >= num_pid_m or n >= num_pid_n:
        return

    # -----------------------------------------------------------------------
    # Block pointers
    # -----------------------------------------------------------------------
    # A: row = m*BLOCK_M, col = 0..K, but we'll loop over K
    offs_am = m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------------------
    # Accumulator (fp32)
    # -----------------------------------------------------------------------
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # -----------------------------------------------------------------------
    # K loop
    # -----------------------------------------------------------------------
    for k in range(0, K, BLOCK_K):
        # Load tiles
        # Use eviction hints for streaming loads
        a = tl.load(a_ptrs, cache_modifier=".evict_first")
        b = tl.load(b_ptrs, cache_modifier=".evict_first")
        # Accumulate in fp32
        acc += tl.dot(a, b, input_precision="ieee")
        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # -----------------------------------------------------------------------
    # Store output (convert back to fp16)
    # -----------------------------------------------------------------------
    offs_cm = m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    tl.store(c_ptrs, acc.to(tl.float16), cache_modifier=".streaming")


# ---------------------------------------------------------------------------
# Wrapper that launches the kernel
# ---------------------------------------------------------------------------
def triton_run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.dtype == torch.float16 and b.dtype == torch.float16
    assert a.shape == (4096, 4096) and b.shape == (4096, 4096)
    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb
    # Output in fp16
    c = torch.empty((M, N), dtype=torch.float16, device=a.device)

    # Launch configuration (tunable constants)
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    GROUP_M = 8
    num_warps = 8
    num_stages = 4

    # Grid size: number of programs = number of (M blocks) * (N blocks)
    grid_m = triton.cdiv(M, BLOCK_M)
    grid_n = triton.cdiv(N, BLOCK_N)
    grid = (grid_m * grid_n,)

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