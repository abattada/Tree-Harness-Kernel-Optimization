import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
import math

# ---------------------------------------------------------------------------
# Triton kernel for fused int4 dequantisation + matrix multiplication
#
# w_packed shape: (K // 8, N)  int32, each element holds 8 int4 values
#   bits [4*i : 4*i+4] correspond to weight row 8*block_row + i.
# Scales shape: (N,)  fp16
# ---------------------------------------------------------------------------

@triton.jit
def _int4_gemm_kernel(
    a_ptr,                         # (M, K)   fp16
    w_packed_ptr,                  # (K // 8, N) int32
    scales_ptr,                    # (N,)     fp16
    out_ptr,                       # (M, N)   fp16
    M, K, N,
    stride_a_m, stride_a_k,
    stride_wp_k8, stride_wp_n,
    stride_scales_n,
    stride_out_m, stride_out_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,         # must be multiple of 8
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

    BLOCK_K_PACKED = BLOCK_K // 8

    # Loop over K dimension
    for k_start in range(0, K, BLOCK_K):
        # Load a tile (BLOCK_M, BLOCK_K) from global
        a_ptrs = a_ptr + (offs_m[:, None] * stride_a_m + (k_start + tl.arange(0, BLOCK_K)[None, :]) * stride_a_k)
        a_tile = tl.load(a_ptrs)   # (BLOCK_M, BLOCK_K)

        # Load packed weight tile (BLOCK_K_PACKED, BLOCK_N)
        packed_k_start = k_start // 8
        wp_ptrs = w_packed_ptr + (packed_k_start + tl.arange(0, BLOCK_K_PACKED)[:, None]) * stride_wp_k8 + offs_n[None, :] * stride_wp_n
        packed_tile = tl.load(wp_ptrs)   # (BLOCK_K_PACKED, BLOCK_N)

        # Load scales tile (BLOCK_N,)
        scales_ptrs = scales_ptr + offs_n * stride_scales_n
        scales_tile = tl.load(scales_ptrs).to(tl.float16)   # (BLOCK_N,)

        # Unpack the 8 int4 values and accumulate dot products
        for i in range(8):
            shift = 4 * i
            # Extract i-th int4 from each packed element
            w_slice_int = (packed_tile >> shift) & 0xF   # (BLOCK_K_PACKED, BLOCK_N)
            w_slice_fp = w_slice_int.to(tl.float16) - 8.0
            w_slice_fp = w_slice_fp * scales_tile[None, :]   # (BLOCK_K_PACKED, BLOCK_N)

            # Corresponding columns of a_tile: columns i, i+8, i+16, ...
            col_start = i
            col_indices = col_start + tl.arange(0, BLOCK_K_PACKED) * 8   # (BLOCK_K_PACKED,)
            # Load a_block for these columns from global (we cannot index a_tile non‑contiguously)
            a_block_ptrs = a_ptr + (offs_m[:, None] * stride_a_m + (k_start + col_indices[None, :]) * stride_a_k)
            a_block = tl.load(a_block_ptrs)   # (BLOCK_M, BLOCK_K_PACKED)

            # Dot product: (BLOCK_M, BLOCK_K_PACKED) x (BLOCK_K_PACKED, BLOCK_N) -> (BLOCK_M, BLOCK_N)
            acc += tl.dot(a_block.to(tl.float32), w_slice_fp.to(tl.float32))

    # Store output tile
    out_ptrs = out_ptr + (offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n)
    tl.store(out_ptrs, acc.to(tl.float16))


def triton_run(a, w_packed, scales):
    M, K = a.shape
    N = scales.shape[0]

    # Block sizes – all divide 4096 exactly
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64  # must be a multiple of 8

    grid = lambda meta: ( (M // BLOCK_M) * (N // BLOCK_N), )

    out = torch.empty((M, N), device=a.device, dtype=torch.float16)

    _int4_gemm_kernel[grid](
        a, w_packed, scales, out,
        M, K, N,
        a.stride(0), a.stride(1),
        w_packed.stride(0), w_packed.stride(1),
        scales.stride(0),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=8,
        num_stages=2,
    )
    return out