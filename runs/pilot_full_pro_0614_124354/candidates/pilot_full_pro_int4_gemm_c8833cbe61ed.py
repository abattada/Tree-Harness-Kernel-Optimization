import torch
import triton
import triton.language as tl


GROUP_K = 8


def _int4_gemm_kernel(
    A_ptr,
    W_packed_ptr,
    scales_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_wk,
    stride_wn,
    stride_s,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Accumulates C = A @ dequantized(W) tile by tile.
    Each program handles a BLOCK_M x BLOCK_N tile of the output.
    The K dimension is traversed in blocks of BLOCK_K, and within each
    block the 4-bit weights are unpacked in groups of 8 (one int32 per 8 rows).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row and column ranges for this tile
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm_mask = rm < M
    rn_mask = rn < N

    # Load the per-column scales (constant over K)
    scales_ptrs = scales_ptr + rn * stride_s
    scales_row = tl.load(scales_ptrs, mask=rn_mask, other=0.0)  # fp16

    # Accumulator in fp32 for precision
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Offsets to extract the 8 nibbles of an int32
    # tl.arange(0, 32, 4) is not supported; use step-by-multiply
    shifts = tl.arange(0, 8) * 4  # [0, 4, 8, 12, 16, 20, 24, 28]

    for k_start in range(0, K, BLOCK_K):
        for k_micro in range(0, BLOCK_K, GROUP_K):
            k = k_start + k_micro
            k_valid = k + tl.arange(0, GROUP_K)

            # Load A tile: shape (BLOCK_M, GROUP_K), fp16
            a_ptrs = A_ptr + rm[:, None] * stride_am + k_valid[None, :] * stride_ak
            a_tile = tl.load(
                a_ptrs,
                mask=(rm_mask[:, None] & (k_valid[None, :] < K)),
                other=0.0,
            )

            # Load one row of packed int4 weights (int32)
            w_row_idx = k // GROUP_K
            w_row_ptrs = W_packed_ptr + w_row_idx * stride_wk + rn * stride_wn
            w_packed_row = tl.load(w_row_ptrs, mask=rn_mask, other=0)

            # Unpack the 8 nibbles into a (GROUP_K, BLOCK_N) fp16 tile
            nibbles = (w_packed_row[None, :] >> shifts[:, None]) & 0xF
            nibbles_fp32 = nibbles.to(tl.float32)
            scales_fp32 = scales_row.to(tl.float32)[None, :]
            w_tile_fp32 = (nibbles_fp32 - 8.0) * scales_fp32
            w_tile = w_tile_fp32.to(tl.float16)

            # Matmul accumulate for this micro-step
            acc += tl.dot(a_tile, w_tile)

    # Store the result tile as fp16
    c_ptrs = C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=(rm_mask[:, None] & rn_mask[None, :]))


# Apply JIT without the decorator to avoid static checker false positives
_int4_gemm_kernel = triton.jit(_int4_gemm_kernel)


def triton_run(a: torch.Tensor, w_packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """
    a:        float16 [M, K]   (4096, 4096)
    w_packed: int32   [K//8, N] (512, 4096)
    scales:   float16 [N]       (4096,)
    returns:  float16 [M, N]    (4096, 4096)
    """
    M, K = a.shape
    _, N = w_packed.shape
    assert K % 8 == 0, "K must be divisible by 8"

    C = torch.empty((M, N), dtype=a.dtype, device=a.device)

    # Tiling tuned for Blackwell: good occupancy with large tiles, pipelined loads
    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 128

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _int4_gemm_kernel[grid](
        a,
        w_packed,
        scales,
        C,
        M, N, K,
        a.stride(0), a.stride(1),
        w_packed.stride(0), w_packed.stride(1),
        scales.stride(0),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=8,
        num_stages=3,
    )

    return C