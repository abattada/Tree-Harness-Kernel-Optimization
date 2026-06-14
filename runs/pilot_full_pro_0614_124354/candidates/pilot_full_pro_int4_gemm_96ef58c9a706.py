import torch
import triton
import triton.language as tl


@triton.jit
def _int4_gemm_kernel(
    a_ptr, w_packed_ptr, scales_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,       # a [M, K] row‑major
    stride_wk, stride_wn,       # w_packed [K//8, N] row‑major
    stride_cm, stride_cn,       # c [M, N] row‑major
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    # scales shared across the whole K loop
    scales = tl.load(scales_ptr + offs_n, mask=mask_n, other=0.0)

    # accumulate in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        # ---- A tile ----
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am
                          + (k_start + offs_k[None, :]) * stride_ak)
        a_mask = mask_m[:, None] & ((k_start + offs_k[None, :]) < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # ---- w_packed tile ----
        k_packed_start = k_start // 8
        offs_kp = k_packed_start + tl.arange(0, BLOCK_K // 8)
        w_ptrs = w_packed_ptr + (offs_kp[:, None] * stride_wk
                                 + offs_n[None, :] * stride_wn)
        w_mask = (offs_kp[:, None] < (K // 8)) & mask_n[None, :]
        w_int = tl.load(w_ptrs, mask=w_mask, other=0)

        # Unpack 4‑bit values: each int32 → 8 float16 elements
        # shapes: w_int (BLOCK_K//8, BLOCK_N)
        shifts = tl.arange(0, 8)[:, None, None] * 4          # (8, 1, 1)
        w_4b = (w_int[None, :, :] >> shifts) & 0xF            # (8, BLOCK_K//8, BLOCK_N)
        w_4b = w_4b.reshape((BLOCK_K, BLOCK_N))               # (BLOCK_K, BLOCK_N)
        w_fp16 = (w_4b.to(tl.float16) - 8.0) * scales[None, :]

        # ---- matmul tile ----
        acc += tl.dot(a, w_fp16)

    # ---- store ----
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm
                      + offs_n[None, :] * stride_cn)
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


def triton_run(a, w_packed, scales):
    """
    a:        fp16 [4096, 4096]
    w_packed: int32 [512, 4096]
    scales:   fp16 [4096]
    returns:  fp16 [4096, 4096]
    """
    M, K = a.shape
    N = scales.shape[0]
    c = torch.empty((M, N), dtype=torch.float16, device=a.device)

    # tunable block sizes (all dividing exactly full dimensions 4096)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 128
    assert BLOCK_K % 8 == 0

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _int4_gemm_kernel[grid](
        a, w_packed, scales, c,
        M, N, K,
        a.stride(0), a.stride(1),
        w_packed.stride(0), w_packed.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return c