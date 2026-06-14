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
    """Triton kernel for int4 GEMM: C = A @ (unpack(W) * scales)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load scaling factors for these N columns (one per output column)
    scales_ptrs = scales_ptr + rn * stride_s
    scales_row = tl.load(scales_ptrs)  # fp16, shape (BLOCK_N,)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    NUM_PACKED = BLOCK_K // GROUP_K  # number of packed rows inside one K‑block

    # Loop over K dimension in blocks of BLOCK_K
    for k_start in range(0, K, BLOCK_K):
        # Load a tile of A (BLOCK_M, BLOCK_K)
        rk = k_start + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
        a_tile = tl.load(a_ptrs)  # fp16, (BLOCK_M, BLOCK_K)

        # Process each packed row (contains 8 K elements)
        for g in tl.static_range(NUM_PACKED):
            # Index of the packed int32 row in W_packed
            w_row_idx = k_start // GROUP_K + g
            w_packed_ptrs = W_packed_ptr + w_row_idx * stride_wk + rn * stride_wn
            w_packed_row = tl.load(w_packed_ptrs)  # int32, (BLOCK_N,)

            # Dequantize one packed row into a small (8, BLOCK_N) weight tile
            w_tile_small = tl.zeros((GROUP_K, BLOCK_N), dtype=tl.float16)
            for j in tl.static_range(GROUP_K):
                nibble = (w_packed_row >> (4 * j)) & 0xF
                w_val_f32 = (nibble.to(tl.float32) - 8.0) * scales_row.to(tl.float32)
                w_val_f16 = w_val_f32.to(tl.float16)
                # Place the values into row j of the small tile
                row_mask = tl.arange(0, GROUP_K)[:, None] == j
                w_val_bcast = w_val_f16[None, :].broadcast_to((GROUP_K, BLOCK_N))
                w_tile_small = tl.where(row_mask, w_val_bcast, w_tile_small)

            # Corresponding slice of A (BLOCK_M, 8)
            a_slice = a_tile[:, g * GROUP_K : (g + 1) * GROUP_K]

            # Accumulate the matmul tile: (BLOCK_M, 8) @ (8, BLOCK_N) -> (BLOCK_M, BLOCK_N)
            acc += tl.dot(a_slice, w_tile_small)

    # Write result to C
    c_ptrs = C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(torch.float16))


# Jit the kernel without using the @ decorator to avoid any static checker false positives.
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

    BLOCK_M = 128
    BLOCK_N = 128
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
        num_stages=2,
    )

    return C