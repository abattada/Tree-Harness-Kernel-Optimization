import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
import math

# ---------------------------------------------------------------------------
# Triton kernel for fused int4 dequantization + matrix multiplication
# ---------------------------------------------------------------------------
#
# w is stored as int4 in packed int32 format:
#   - w_packed shape: (K // 8, N)   (K = 4096, N = 4096)
#   - For each element (i, j), the 8 int4 values are stored in bits [4*i : 4*i+4]
#     of w_packed[i, j], corresponding to w[8*i + i', j] for i' = 0..7.
#
# The kernel loads tiles of a (M, K) and tiles of w_packed, unpacks the int4
# values on the fly (with scaling by `scales`), and accumulates the dot product
# into the output.
#
# Block sizes are chosen so that all dimensions are exactly divisible.
# ---------------------------------------------------------------------------

@triton.jit
def _int4_gemm_kernel(
    a_ptr,                         # (M, K)   fp16
    w_packed_ptr,                  # (K // 8, N) int32
    scales_ptr,                    # (N,)     fp16
    out_ptr,                       # (M, N)   fp16
    M, K, N,
    stride_a_m, stride_a_k,        # strides for a (usually row-major)
    stride_wp_k8, stride_wp_n,     # strides for w_packed
    stride_scales_n,               # stride for scales = 1
    stride_out_m, stride_out_n,    # strides for output
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,         # must be multiple of 8
):
    # Program ID and tile start indices
    pid = tl.program_id(0)
    num_n_blocks = tl.num_programs(0)
    pid_m = pid // (N // BLOCK_N)
    pid_n = pid % (N // BLOCK_N)

    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N

    # Offsets for rows and columns inside a tile
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Number of packed rows needed per K-tile
    BLOCK_K_PACKED = BLOCK_K // 8   # number of int32 rows

    # Loop over K dimension
    for k_start in range(0, K, BLOCK_K):
        # Pointer to start of a block: a[start_m:start_m+BLOCK_M, k_start:k_start+BLOCK_K]
        a_ptrs = a_ptr + (offs_m[:, None] * stride_a_m + (k_start + tl.arange(0, BLOCK_K)[None, :]) * stride_a_k)

        # Load a tile (BLOCK_M, BLOCK_K) - no mask needed as sizes divide exactly
        a_tile = tl.load(a_ptrs)

        # Pointer to start of packed block: w_packed[k_start//8 : (k_start+BLOCK_K)//8, start_n:start_n+BLOCK_N]
        packed_k_start = k_start // 8
        wp_ptrs = w_packed_ptr + (packed_k_start + tl.arange(0, BLOCK_K_PACKED)[:, None]) * stride_wp_k8 + offs_n[None, :] * stride_wp_n

        # Load all packed rows for this K-block as int32
        packed_tile = tl.load(wp_ptrs)   # (BLOCK_K_PACKED, BLOCK_N)

        # Load scales tile (BLOCK_N,)
        scales_ptrs = scales_ptr + offs_n * stride_scales_n
        scales_tile = tl.load(scales_ptrs)  # (BLOCK_N,)
        scales_tile = scales_tile.to(tl.float16)

        # Inner loop over the 8 int4 values inside each packed element
        # We unroll by iterating over i = 0..7 and extract the i-th int4.
        for i in range(8):
            shift = i * 4
            # Extract the i-th int4 from each element of packed_tile
            # packed_tile shape: (BLOCK_K_PACKED, BLOCK_N) -> shift produces same shape
            w_slice_int = (packed_tile >> shift) & 0xF   # still on device, int32
            # Convert to float16, subtract 8
            w_slice_fp = w_slice_int.to(tl.float16) - 8.0
            # Broadcast scales across rows (each row multiplied by same scale)
            w_slice_fp = w_slice_fp * scales_tile[None, :]   # (BLOCK_K_PACKED, BLOCK_N)

            # Now obtain the corresponding slice of a: columns [k_start + i, k_start + i + 8, ..., k_start + i + 8*(BLOCK_K_PACKED-1)]
            # Actually a_tile has shape (BLOCK_M, BLOCK_K). The columns we need for this i are those at indices
            # i, i+8, i+16, ... up to i + 8*(BLOCK_K_PACKED-1).
            # We can pick them using indexing with a mask or using tl.reshape.
            # Since BLOCK_K is small (e.g. 64) and we can compute on the fly, we use a simple approach:
            # create a column offset vector of length BLOCK_K_PACKED (the number of rows of w_slice)
            col_start_in_block = i
            strided_idx = col_start_in_block + tl.arange(0, BLOCK_K_PACKED) * 8   # shape (BLOCK_K_PACKED,)
            # Gather these columns from a_tile:
            # a_tile is (BLOCK_M, BLOCK_K). We want rows of a_tile (all rows) and columns indexed by strided_idx.
            # Use pointer arithmetic and tl.load inside the kernel is not allowed; we can use tl.gather? Not available.
            # Alternative: we can compute a_block directly by loading a new slice from global memory.
            # Since BLOCK_K is small, repeating the load inside the inner loop is acceptable.
            # We'll load a_block from global a.
            a_block_ptrs = a_ptr + (offs_m[:, None] * stride_a_m + (k_start + strided_idx[None, :]) * stride_a_k)
            a_block = tl.load(a_block_ptrs)   # (BLOCK_M, BLOCK_K_PACKED)

            # Dot product: accumulate (BLOCK_M, BLOCK_K_PACKED) x (BLOCK_K_PACKED, BLOCK_N) -> (BLOCK_M, BLOCK_N)
            acc += tl.dot(a_block.to(tl.float32), w_slice_fp.to(tl.float32),
                         input_precision='ieee')

    # Store result
    out_ptrs = out_ptr + (offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n)
    tl.store(out_ptrs, acc.to(tl.float16))


def triton_run(a, w_packed, scales):
    """
    a        : fp16 [4096, 4096]
    w_packed : int32 [512, 4096]
    scales   : fp16 [4096]
    returns  : fp16 [4096, 4096]  (a @ w)
    """
    assert a.shape == (4096, 4096)
    assert w_packed.shape == (512, 4096)
    assert scales.shape == (4096,)
    M, K = a.shape
    N = scales.shape[0]

    # Block sizes: all divide 4096 exactly
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64  # divisible by 8

    # Grid size
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