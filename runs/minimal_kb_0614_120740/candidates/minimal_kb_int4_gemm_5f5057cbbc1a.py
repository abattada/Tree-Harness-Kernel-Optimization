import torch
import triton
import triton.language as tl

def _int4_gemm_kernel_def(
    a_ptr, w_packed_ptr, scales_ptr, out_ptr,
    M, K, N,
    stride_a_m, stride_a_k,
    stride_wp_k8, stride_wp_n,
    stride_scales_n,
    stride_out_m, stride_out_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n_blocks = N // BLOCK_N
    pid_m = pid // num_n_blocks
    pid_n = pid % num_n_blocks
    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    pk = BLOCK_K // 8

    for k_start in range(0, K, BLOCK_K):
        # Load a tile of a
        a_ptrs = a_ptr + (offs_m[:, None] * stride_a_m +
                          (k_start + tl.arange(0, BLOCK_K)[None, :]) * stride_a_k)
        a_tile = tl.load(a_ptrs)

        # Load packed weights
        pk_start = k_start // 8
        wp_ptrs = (w_packed_ptr +
                   (pk_start + tl.arange(0, pk)[:, None]) * stride_wp_k8 +
                   offs_n[None, :] * stride_wp_n)
        packed = tl.load(wp_ptrs)

        # Load scales
        scales_ptrs = scales_ptr + offs_n * stride_scales_n
        scales_tile = tl.load(scales_ptrs).to(tl.float32)

        # Unpack 8 int4 values per packed element
        for i in range(8):
            w_int = (packed >> (i * 4)) & 0xF
            w_fp = w_int.to(tl.float32) - 8.0
            w_fp = w_fp * scales_tile[None, :]

            # Corresponding columns in a_tile
            col_idx = k_start + i + 8 * tl.arange(0, pk)
            a_block_ptrs = (a_ptr +
                            offs_m[:, None] * stride_a_m +
                            col_idx[None, :] * stride_a_k)
            a_block = tl.load(a_block_ptrs)

            acc += tl.dot(a_block.to(tl.float32), w_fp)

    out_ptrs = (out_ptr +
                offs_m[:, None] * stride_out_m +
                offs_n[None, :] * stride_out_n)
    tl.store(out_ptrs, acc.to(tl.float16))

_int4_gemm_kernel = triton.jit(_int4_gemm_kernel_def)

def triton_run(a, w_packed, scales):
    M, K = a.shape
    N = scales.shape[0]
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64

    grid = lambda meta: ((M // BLOCK_M) * (N // BLOCK_N),)
    out = torch.empty((M, N), device=a.device, dtype=torch.float16)

    _int4_gemm_kernel[grid](
        a, w_packed, scales, out,
        M, K, N,
        a.stride(0), a.stride(1),
        w_packed.stride(0), w_packed.stride(1),
        scales.stride(0),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=8, num_stages=2,
    )
    return out