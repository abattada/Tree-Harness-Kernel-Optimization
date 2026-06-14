import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton GEMM kernel: C = A @ B   (fp16 inputs, fp16 output, fp32 accumulation)
# ---------------------------------------------------------------------------
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this block (full tile assumed; masks handle leftovers)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)   # reused for each K-step

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Looping over K dimension
    for start_k in range(0, K, BLOCK_K):
        # Load A tile: shape (BLOCK_M, BLOCK_K)
        a_offs = (offs_m[:, None] * stride_am +
                  (start_k + offs_k[None, :]) * stride_ak)
        # Boundary masks: only needed if K not multiple of BLOCK_K,
        # but we include them for correctness.
        a_mask = (offs_m[:, None] < M) & ((start_k + offs_k[None, :]) < K)
        a = tl.load(a_ptr + a_offs, mask=a_mask, other=0.0)

        # Load B tile: shape (BLOCK_K, BLOCK_N)
        b_offs = ((start_k + offs_k[:, None]) * stride_bk +
                  offs_n[None, :] * stride_bn)
        b_mask = ((start_k + offs_k[:, None]) < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptr + b_offs, mask=b_mask, other=0.0)

        # Accumulate using tensor core (fp16 inputs, fp32 accumulator)
        acc += tl.dot(a, b)

    # Convert result to fp16 and store
    c = acc.to(tl.float16)
    c_offs = offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptr + c_offs, c, mask=c_mask)


# ---------------------------------------------------------------------------
# triton_run: allocates output and launches the kernel
# ---------------------------------------------------------------------------
def triton_run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda
    assert a.dtype == torch.float16 and b.dtype == torch.float16
    M, K = a.shape
    _, N = b.shape
    assert a.stride(0) == a.size(1) and a.stride(1) == 1  # contiguous A
    assert b.stride(0) == b.size(1) and b.stride(1) == 1  # contiguous B

    c = torch.empty((M, N), device='cuda', dtype=torch.float16)

    # Launch configuration – favour higher occupancy (smaller blocks, more programs)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),
                         triton.cdiv(N, meta['BLOCK_N']))

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=8,
        num_stages=3,
    )
    return c