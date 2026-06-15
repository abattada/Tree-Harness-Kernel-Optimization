import torch
import triton
import triton.language as tl

@triton.jit
def addmm_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # load bias
    bias = tl.load(bias_ptr + offs_m, mask=offs_m < M, other=0.0).to(tl.float32)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + (offs_k + k0)[None, :] * stride_ak
        b_ptrs = b_ptr + (offs_k + k0)[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & ((offs_k + k0)[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=((offs_k + k0)[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        acc = tl.dot(a, b, acc)

    # add bias
    acc += bias[:, None]
    c = acc.to(tl.float16)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_run(bias: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    N = b.shape[1]
    assert a.shape[1] == b.shape[0] == K
    assert bias.shape[0] == M

    c = torch.empty((M, N), dtype=torch.float16, device=a.device)

    # block sizes tuned for RTX 5090
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
    GROUP_M = 8
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    addmm_kernel[grid](
        a, b, bias, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
        GROUP_M,
        num_warps=4,
        num_stages=2,
    )

    return c