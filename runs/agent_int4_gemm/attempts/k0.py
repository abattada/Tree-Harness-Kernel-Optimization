import torch
import triton
import triton.language as tl

# int4 W4A16 GEMM: a[M,K] f16 @ w[K,N] where w is int4 packed along K into int32
# (w_packed[K//8, N]), dequant = (nibble - 8) * scales[n].
# scales is per-N -> factored OUT of the K loop and applied once to the f32
# accumulator (scales[n] * sum_k a*(w_int-8)). Hot loop does only f16 tensor-core
# matmul + cheap shift/mask dequant.

# Grid rationale (compute-bound 4096^3 on RTX 5090 / Blackwell sm_120):
#   BLOCK_M/N in {64,128,256}: tensor-core friendly (mult of 16), trade L2 reuse
#     vs register/SMEM pressure. BLOCK_K=64: 8x packed rows -> 8 packed int32 rows
#     per step, good MMA-K depth without blowing SMEM.
#   num_warps 4/8, num_stages 3/4: software-pipeline the global loads of a/w.
#   GROUP_M=8: L2 swizzle so column-tiles reuse the same a rows.
configs = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=4),
]


@triton.autotune(configs=configs, key=["M", "N", "K"])
@triton.jit
def _int4_gemm_kernel(
    a_ptr, w_ptr, scales_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wk, stride_wn,
    stride_cm, stride_cn,
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
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_wn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    # redundant-load trick: row k maps to packed row k//8 (each group of 8 K-rows
    # loads the SAME int32, L2-cached), then shift by 4*(k%8) to pick the nibble.
    w_row = offs_k // 8
    w_ptrs = w_ptr + w_row[:, None] * stride_wk + offs_wn[None, :] * stride_wn
    shifter = (offs_k % 8) * 4

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        wp = tl.load(w_ptrs)
        w = ((wp >> shifter[:, None]) & 0xF) - 8
        w = w.to(tl.float16)
        acc = tl.dot(a, w, acc)
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += (BLOCK_K // 8) * stride_wk

    scales = tl.load(scales_ptr + offs_wn).to(tl.float32)
    acc = acc * scales[None, :]
    c = acc.to(tl.float16)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def triton_run(a, w_packed, scales):
    M, K = a.shape
    N = w_packed.shape[1]
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    _int4_gemm_kernel[grid](
        a, w_packed, scales, c,
        M, N, K,
        a.stride(0), a.stride(1),
        w_packed.stride(0), w_packed.stride(1),
        c.stride(0), c.stride(1),
    )
    return c
