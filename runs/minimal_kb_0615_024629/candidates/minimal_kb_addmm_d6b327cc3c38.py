import torch
import triton
import triton.language as tl

@triton.jit
def addmm_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row and column indices for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # K index within the inner loop
    offs_k = tl.arange(0, BLOCK_K)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Load bias once per program (broadcast over columns)
    bias = tl.load(bias_ptr + offs_m, mask=offs_m < M)  # mask for safety even though M is multiple
    bias = bias.to(tl.float32)

    for k0 in range(0, K, BLOCK_K):
        # Pointers for A tile (M x BLOCK_K)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + (offs_k + k0)[None, :] * stride_ak
        # Pointers for B tile (BLOCK_K x N)
        b_ptrs = b_ptr + (offs_k + k0)[:, None] * stride_bk + offs_n[None, :] * stride_bn

        # Load tiles (all in range because dimensions are multiples)
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)

        acc += tl.dot(a, b)

    # Add bias (broadcast across columns)
    acc += bias[:, None]

    # Convert to fp16 and store
    c = acc.to(tl.float16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c)


def triton_run(bias: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    N = b.shape[1]
    assert a.shape[1] == b.shape[0] == K
    assert bias.shape[0] == M

    # Output allocation
    c = torch.empty((M, N), dtype=torch.float16, device=a.device)

    # Launch configuration (tuned for RTX 5090, high occupancy)
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    addmm_kernel[grid](
        a, b, bias, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
        num_warps=4,
        num_stages=2,
    )

    return c