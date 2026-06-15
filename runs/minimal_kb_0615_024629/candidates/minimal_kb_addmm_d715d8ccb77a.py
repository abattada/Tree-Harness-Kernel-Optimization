import torch
import triton
import triton.language as tl
import math

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 4}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.heuristics({
    'EVEN_M': lambda args: args['M'] % args['BLOCK_M'] == 0,
    'EVEN_N': lambda args: args['N'] % args['BLOCK_N'] == 0,
    'EVEN_K': lambda args: args['K'] % args['BLOCK_K'] == 0,
})
@triton.jit
def addmm_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_K: tl.constexpr,
):
    # Swizzle program IDs for better L2 cache reuse
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Tile start indices
    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N

    # Offsets for the first tile in K dimension
    offs_k = tl.arange(0, BLOCK_K)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K tiles
    for k in range(0, K, BLOCK_K):
        # Load A tile: (BLOCK_M, BLOCK_K) fp16
        if EVEN_K:
            a_mask = None
        else:
            a_mask = offs_k[None, :] < (K - k)
        a = tl.load(a_ptr + (start_m[:, None] * stride_am + (offs_k[None, :] + k) * stride_ak),
                    mask=a_mask, other=0.0)
        # Load B tile: (BLOCK_K, BLOCK_N) fp16
        if EVEN_K:
            b_mask = None
        else:
            b_mask = offs_k[:, None] < (K - k)
        b = tl.load(b_ptr + ((offs_k[:, None] + k) * stride_bk + start_n[None, :] * stride_bn),
                    mask=b_mask, other=0.0)
        # Dot product accumulate
        acc += tl.dot(a, b)

    # Load bias: (BLOCK_N,) broadcast across rows
    offs_n = tl.arange(0, BLOCK_N)
    bias_ptrs = bias_ptr + start_n + offs_n
    if EVEN_N:
        bias = tl.load(bias_ptrs)
    else:
        bias = tl.load(bias_ptrs, mask=offs_n < N - start_n, other=0.0)

    # Add bias and convert back to fp16
    c = (acc + bias[None, :]).to(tl.float16)

    # Store output
    offs_m = tl.arange(0, BLOCK_M)
    c_ptrs = c_ptr + (start_m[:, None] * stride_cm + start_n[None, :] * stride_cn)
    if EVEN_M and EVEN_N:
        tl.store(c_ptrs, c)
    else:
        m_mask = offs_m[:, None] < (M - start_m)
        n_mask = offs_n[None, :] < (N - start_n)
        tl.store(c_ptrs, c, mask=m_mask & n_mask)


def triton_run(bias: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute bias + a @ b using Triton.
    All tensors are fp16 on CUDA.
    """
    assert bias.dtype == torch.float16, "bias must be fp16"
    assert a.dtype == torch.float16, "a must be fp16"
    assert b.dtype == torch.float16, "b must be fp16"
    assert a.shape[1] == b.shape[0], "inner dims must match"
    assert bias.shape[0] == b.shape[1], "bias dim must match N"

    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    # Output tensor
    c = torch.empty((M, N), dtype=torch.float16, device=a.device)

    # Grid launch
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

    addmm_kernel[grid](
        a, b, bias, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c