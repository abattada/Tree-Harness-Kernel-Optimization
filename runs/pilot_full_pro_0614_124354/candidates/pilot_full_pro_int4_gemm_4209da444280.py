import torch
import triton
import triton.language as tl


@triton.jit
def _int4_gemm_kernel(
    a_ptr, w_packed_ptr, scales_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wk, stride_wn,
    stride_sc,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Compute C = A @ W where W is dequantized from packed int4 weights.
    A: fp16 [M, K], W_packed: int32 [K//8, N], scales: fp16 [N]
    C: fp16 [M, N]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for the C tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for A and C
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn

    # Load scales for this N‑tile once
    scales = tl.load(scales_ptr + offs_n * stride_sc)  # [BLOCK_N]

    # Accumulator for C tile (fp32 for precision)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K loop
    for k in range(0, K, BLOCK_K):
        # Load A tile [BLOCK_M, BLOCK_K] fp16
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak)

        # Load packed W tile [BLOCK_K//8, BLOCK_N] int32
        k_pack_start = k // 8
        offs_k_pack = tl.arange(0, BLOCK_K // 8)
        w_ptrs = (w_packed_ptr
                  + (k_pack_start + offs_k_pack[:, None]) * stride_wk
                  + offs_n[None, :] * stride_wn)
        w_pack = tl.load(w_ptrs)  # int32

        # ------------------------------------------------------------------
        # Dequantize: each int32 becomes 8 fp16 values (K dimension).
        # w_pack shape: [BLOCK_K//8, BLOCK_N]
        # shifts: [1, 8, 1]  with values 0,4,8,...,28
        shifts = (tl.arange(0, 8, dtype=tl.int32) * 4)[None, :, None]
        w_expand = w_pack[:, None, :]                          # [BLOCK_K//8, 1, BLOCK_N]
        nibbles = (w_expand >> shifts) & 0xF                   # [BLOCK_K//8, 8, BLOCK_N] int32
        w_fp = tl.cast(nibbles, tl.float16) - 8.0              # [BLOCK_K//8, 8, BLOCK_N] fp16
        w_fp = w_fp * scales[None, None, :]                    # broadcast scales
        w = tl.reshape(w_fp, (BLOCK_K, BLOCK_N))               # [BLOCK_K, BLOCK_N] fp16
        # ------------------------------------------------------------------

        # Tensor core matmul
        acc += tl.dot(a, w)  # a: [BLOCK_M, BLOCK_K], w: [BLOCK_K, BLOCK_N]

    # Store output
    c = acc.to(tl.float16)
    tl.store(c_ptrs, c)


def triton_run(a, w_packed, scales):
    """
    int4_gemm: dequantize w_packed and compute a @ w.
    Args:
        a:          fp16 [4096, 4096]
        w_packed:   int32 [512, 4096]
        scales:     fp16 [4096]
    Returns:
        fp16 [4096, 4096]
    """
    M, K = a.shape
    N = scales.shape[0]
    assert w_packed.shape == (K // 8, N), "w_packed shape mismatch"
    assert scales.shape == (N,), "scales shape mismatch"

    # Output allocation
    c = torch.empty((M, N), dtype=torch.float16, device=a.device)

    # Tuning knobs – chosen to fully utilise the SM without masks for 4096
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 256       # multiple of 8 and 16
    num_warps = 8
    num_stages = 1      # simple loop; can be pipelined in later rounds

    # 2D grid
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _int4_gemm_kernel[grid](
        a, w_packed, scales, c,
        M, N, K,
        a.stride(0), a.stride(1),
        w_packed.stride(0), w_packed.stride(1),
        scales.stride(0),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return c