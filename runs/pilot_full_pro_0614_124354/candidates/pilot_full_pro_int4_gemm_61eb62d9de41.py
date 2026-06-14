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
    Compute C = A * W, where W is dequantized from packed int4 weights.
    A: fp16 [M, K]
    W_packed: int32 [K//8, N]
    scales: fp16 [N]
    C: fp16 [M, N]
    """
    # BLOCK_K must be a multiple of 8 (for int4 packing) and of 16 (for tensor cores)
    tl.static_assert(BLOCK_K % 8 == 0)
    tl.static_assert(BLOCK_K % 16 == 0)

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Load scales for the current N tile once
    scales = tl.load(scales_ptr + offs_n * stride_sc)  # [BLOCK_N]

    # Base pointers for A and C tiles
    a_tile_ptr = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn

    # Accumulator in fp32 for precision
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Shifts for unpacking the 8 nibbles inside each int32 word
    # (values: 0, 4, 8, ..., 28) – constant for the whole kernel
    shifts = tl.arange(0, 32, 4, dtype=tl.int32)[None, :, None]

    for k in range(0, K, BLOCK_K):
        # Load A tile [BLOCK_M, BLOCK_K]
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak)

        # Load packed W tile [BLOCK_K//8, BLOCK_N] as int32
        k_pack_start = k // 8
        offs_k_pack = tl.arange(0, BLOCK_K // 8)
        w_ptrs = (w_packed_ptr
                  + (k_pack_start + offs_k_pack[:, None]) * stride_wk
                  + offs_n[None, :] * stride_wn)
        w_pack = tl.load(w_ptrs)

        # Dequantize: each int32 -> 8 fp16 values (nibbles 0..15 -> -8..7)
        w_expand = w_pack[:, None, :]                            # [BLOCK_K//8, 1, BLOCK_N]
        nibbles = (w_expand >> shifts) & 0xF                     # [BLOCK_K//8, 8, BLOCK_N] int32
        w_fp = tl.cast(nibbles, tl.float16) - 8.0
        w_fp = w_fp * scales[None, None, :]                     # broadcast scales
        w = tl.reshape(w_fp, (BLOCK_K, BLOCK_N))                # [BLOCK_K, BLOCK_N] fp16

        # Tensor-core matrix multiply
        acc += tl.dot(a, w)

    # Store output as fp16
    c = acc.to(tl.float16)
    tl.store(c_ptrs, c)


def triton_run(a, w_packed, scales):
    """
    int4_gemm: dequantize w_packed and multiply with a.

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

    c = torch.empty((M, N), dtype=torch.float16, device=a.device)

    # Tiled launch – all dimensions are multiples of the chosen tile sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 256
    num_warps = 8
    num_stages = 3

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