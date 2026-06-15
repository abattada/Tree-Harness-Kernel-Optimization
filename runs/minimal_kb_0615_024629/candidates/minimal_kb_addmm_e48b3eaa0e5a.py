import torch
import triton
import triton.language as tl

# Triton kernel for addmm: output = bias + A @ B
# A, B, bias are fp16, output is fp16
# Accumulate in fp32 for precision
@triton.jit
def addmm_kernel(
    A_ptr, B_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program ids along M and N dimensions
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offset for the tile in the output and input matrices
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for A and B tiles (we'll iterate over K dimension)
    # A is [M, K] with strides (stride_am, stride_ak)
    # B is [K, N] with strides (stride_bk, stride_bn)
    # We compute dot product tile by tile

    # Set up pointers for the current tile row/col
    A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K loop
    for start_k in range(0, K, BLOCK_K):
        # Load tiles (use masks to handle boundaries, though K is multiple of BLOCK_K)
        a_tile = tl.load(A_ptrs, mask=offs_k[None, :] < K - start_k, other=0.0)
        b_tile = tl.load(B_ptrs, mask=offs_k[:, None] < K - start_k, other=0.0)
        # Compute partial dot product
        acc += tl.dot(a_tile, b_tile, out_dtype=tl.float32)

        # Advance pointers (move to next K block)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    # Load bias (broadcasted across columns)
    # bias is [M] with stride 1
    bias_vals = tl.load(bias_ptr + offs_m, mask=offs_m < M, other=0.0)
    # Add bias onto accumulator column-wise
    acc += bias_vals[:, None].to(tl.float32)

    # Store output (convert to fp16)
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc.to(tl.float16), mask=offs_m[:, None] < M and offs_n[None, :] < N)


def triton_run(bias: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert bias.is_cuda and a.is_cuda and b.is_cuda
    assert bias.dtype == torch.float16 and a.dtype == torch.float16 and b.dtype == torch.float16
    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb
    assert bias.shape == (M,)

    # Allocate output
    out = torch.empty((M, N), dtype=torch.float16, device=a.device)

    # Choose block sizes (tunable parameters)
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    # Grid for 2D tile
    grid = (M // BLOCK_M, N // BLOCK_N)

    # Launch kernel
    addmm_kernel[grid](
        a, b, bias, out,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
        num_warps=8,
        num_stages=4,
    )
    return out