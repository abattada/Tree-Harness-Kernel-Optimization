import torch
import triton
import triton.language as tl

@triton.jit
def _int4_gemm_kernel(
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
    # Compute tile indices
    pid = tl.program_id(0)
    num_n_blocks = N // BLOCK_N
    pid_m = pid // num_n_blocks
    pid_n = pid % num_n_blocks
    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)

    # Accumulator (fp32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Packed K dimension: K // 8
    pk = BLOCK_K // 8

    # Main K loop
    for k_start in range(0, K, BLOCK_K):
        # Load a tile: (BLOCK_M, BLOCK_K)
        a_ptrs = a_ptr + (offs_m[:, None] * stride_a_m + (k_start + tl.arange(0, BLOCK_K)[None, :]) * stride_a_k)
        a_tile = tl.load(a_ptrs)   # no mask needed for exact divisions

        # Load packed tile: (pk, BLOCK_N) int32
        pk_start = k_start // 8
        wp_ptrs = w_packed_ptr + (pk_start + tl.arange(0, pk)[:, None]) * stride_wp_k8 + offs_n[None, :] * stride_wp_n
        packed = tl.load(wp_ptrs)

        # Load scales: (BLOCK_N,) fp16 -> fp32 for later multiplication
        scales_ptrs = scales_ptr + offs_n * stride_scales_n
        scales_tile = tl.load(scales_ptrs).to(tl.float32)

        # Unroll the 8 int4 values within each packed element
        for i in range(8):
            # Extract i-th 4-bit field, shift to 0..15, convert to float32, subtract 8
            w_int = (packed >> (i * 4)) & 0xF
            w_fp = w_int.to(tl.float32) - 8.0   # (pk, BLOCK_N)
            w_fp = w_fp * scales_tile[None, :]  # scale

            # Corresponding columns in a_tile: indices i, i+8, i+16, ...
            col_idx = k_start + i + 8 * tl.arange(0, pk)   # shape (pk,)
            # Load sub-block of a: (BLOCK_M, pk)
            a_block_ptrs = a_ptr + (offs_m[:, None] * stride_a_m + col_idx[None, :] * stride_a_k)
            a_block = tl.load(a_block_ptrs)

            # Dot product accumulate
            acc += tl.dot(a_block.to(tl.float32), w_fp)

    # Store output
    out_ptrs = out_ptr + (offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n)
    tl.store(out_ptrs, acc.to(tl.float16))

def triton_run(a, w_packed, scales):
    M, K = a.shape
    N = scales.shape[0]
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64   # must be multiple of 8, divides K exactly

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