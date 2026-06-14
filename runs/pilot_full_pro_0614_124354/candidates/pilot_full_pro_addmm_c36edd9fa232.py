import torch
import triton
import triton.language as tl


@triton.jit
def addmm_kernel(
    bias_ptr,
    a_ptr,
    b_ptr,
    out_ptr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program ID: tile row (M) and tile column (N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to the first elements of the A and B tiles
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main K loop
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)  # [BLOCK_M, BLOCK_K] f16
        b = tl.load(b_ptrs)  # [BLOCK_K, BLOCK_N] f16

        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

        # Advance pointers along K
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add bias (broadcast across rows)
    bias_offs = offs_n  # length BLOCK_N
    bias_ptrs = bias_ptr + bias_offs
    bias_vals = tl.load(bias_ptrs)                # f16
    acc = acc + bias_vals[None, :].to(tl.float32) # broadcast across BLOCK_M

    # Store output tile
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty))


def triton_run(bias: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert bias.shape == (N,), f"bias shape mismatch: {bias.shape} vs (N={N})"
    assert K == K2, f"Inner dimensions mismatch: {K} vs {K2}"

    # Allocate output
    out = torch.empty((M, N), dtype=a.dtype, device=a.device)

    # Launch configuration (tunable)
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    num_warps = 8
    num_stages = 4

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    addmm_kernel[grid](
        bias,
        a,
        b,
        out,
        stride_am=a.stride(0),
        stride_ak=a.stride(1),
        stride_bk=b.stride(0),
        stride_bn=b.stride(1),
        M=M,
        N=N,
        K=K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return out