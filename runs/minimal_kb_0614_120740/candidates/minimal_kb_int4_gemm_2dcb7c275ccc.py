import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel for int4 GEMM with group-swizzled grid
#
# a: (M, K) float16
# w_packed: (K//8, N) int32  (each int32 packs 8 int4 values)
# scales: (N,) float16
# c: (M, N) float16 output
#
# Dequantization is fused inside the K loop.
# ---------------------------------------------------------------------------
@triton.jit
def int4_gemm_kernel(
    a_ptr, w_ptr, scales_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wk, stride_wn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # ---- Group-swizzled program ID mapping ----
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

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K loop
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # ---- Load A tile ----
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=a_mask,
            other=0.0,
        )

        # ---- Load packed weight tile ----
        w_offs_row = offs_k // 8
        w_mask = (w_offs_row[:, None] < (K // 8)) & (offs_n[None, :] < N)
        w = tl.load(
            w_ptr + w_offs_row[:, None] * stride_wk + offs_n[None, :] * stride_wn,
            mask=w_mask,
            other=0,
        )

        # ---- Load scales ----
        s = tl.load(scales_ptr + offs_n, mask=offs_n < N, other=0.0)

        # ---- Dequantize: unpack nibbles from int32 ----
        w_3d = w[:, None, :]                           # (BLOCK_K//8, 1, BLOCK_N)
        shifts = tl.arange(0, 8) * 4                  # (8,)
        nibbles = (w_3d >> shifts[None, :, None]) & 0xF   # (BLOCK_K//8, 8, BLOCK_N)
        w_fp = tl.reshape(nibbles, (BLOCK_K, BLOCK_N)).to(tl.float16)
        w_fp = (w_fp - 8.0) * s[None, :]

        # ---- Dot product (fp16 inputs, fp32 accumulate) ----
        acc += tl.dot(a.to(tl.float16), w_fp)

    # ---- Store result ----
    c = acc.to(tl.float16)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        c,
        mask=c_mask,
    )


# ---------------------------------------------------------------------------
# Public API: triton_run(a, w_packed, scales) -> output
# ---------------------------------------------------------------------------
def triton_run(a: torch.Tensor, w_packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    assert w_packed.shape[0] == K // 8
    N = w_packed.shape[1]
    assert scales.shape == (N,)

    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # Block sizes chosen for NVIDIA Blackwell (RTX 5090)
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    GROUP_M = 8

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

    int4_gemm_kernel[grid](
        a, w_packed, scales, c,
        M, N, K,
        a.stride(0), a.stride(1),
        w_packed.stride(0), w_packed.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
        GROUP_M,
        num_warps=8,
        num_stages=4,
        num_ctas=1,
    )

    return c