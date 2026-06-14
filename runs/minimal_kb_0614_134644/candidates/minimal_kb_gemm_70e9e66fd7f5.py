import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Grid program IDs
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    # Group ordering for L2 reuse
    group_id = pid // num_pid_n
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_n) // group_size_m

    # Tile start indices
    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N

    # Offsets in M and N dimensions
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to A and B tiles (with proper masks)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Tile K iteration
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Compute tile boundaries for this K chunk
        k_start = k * BLOCK_K
        offs_k_chunk = k_start + tl.arange(0, BLOCK_K)
        # Mask loads (handy in case of non-divisible sizes)
        a_mask = (offs_m[:, None] < M) & (offs_k_chunk[None, :] < K)
        b_mask = (offs_k_chunk[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        # Compute dot product
        acc = tl.dot(a, b, acc)
        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Convert to fp16 and store
    c = acc.to(tl.float16)
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == (4096, 4096) and b.shape == (4096, 4096)
    assert a.dtype == b.dtype == torch.float16
    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb
    c = torch.empty((M, N), dtype=torch.float16, device=a.device)

    # Tuned for RTX 5090 (sm_120) with 4096x4096
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    GROUP_M = 8

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        num_warps=8,
        num_stages=3,
    )
    return c