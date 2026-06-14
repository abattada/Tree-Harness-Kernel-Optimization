import torch
import triton
import triton.language as tl

# Each int32 holds eight 4-bit weights (group size = 8).
GROUP_K = 8

@triton.jit
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
    Matrix multiplication C = A * W, where W is dequantized on the fly
    from packed int4.  Each program computes a BLOCK_M x BLOCK_N tile of C.
    The K dimension is blocked by BLOCK_K; inside each block we load the
    corresponding packed-weights and dequantize them into a (BLOCK_K, BLOCK_N)
    tile, then perform a single tl.dot with the A tile.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row / column ranges (no mask needed because M and N divide BLOCK_x).
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load scaling factors for the N columns (one per column).
    scales_ptrs = scales_ptr + rn * stride_s
    scales_row = tl.load(scales_ptrs)  # fp16, shape (BLOCK_N,)

    # Accumulator in fp32.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Number of 8-element groups inside BLOCK_K (integer division).
    NUM_PACKED_ROWS = BLOCK_K // GROUP_K  # = 16 for BLOCK_K=128

    for k_start in range(0, K, BLOCK_K):
        # ------- Load the A tile (BLOCK_M, BLOCK_K) ----------
        rk = k_start + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
        a_tile = tl.load(a_ptrs)  # (BLOCK_M, BLOCK_K)

        # ------- Load all packed-weight rows for this K block ------
        # packed row index corresponding to k_start
        w_row_base = k_start // GROUP_K
        # Load a (NUM_PACKED_ROWS, BLOCK_N) tile of int32.
        w_packed_rows = tl.zeros((NUM_PACKED_ROWS, BLOCK_N), dtype=tl.int32)
        for i in tl.static_range(NUM_PACKED_ROWS):
            w_ptrs = (W_packed_ptr
                      + (w_row_base + i) * stride_wk
                      + rn * stride_wn)
            w_packed_rows = tl.where(
                tl.arange(0, NUM_PACKED_ROWS)[:, None] == i,
                tl.load(w_ptrs),
                w_packed_rows,
            )

        # ------- Dequantize packed rows into W_tile (BLOCK_K, BLOCK_N) -------
        w_tile = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float16)
        for i in tl.static_range(NUM_PACKED_ROWS):
            packed_row = w_packed_rows  # shape (NUM_PACKED_ROWS, BLOCK_N)  -- pick row i
            # We need the i-th row: use a mask.
            row_mask_2d = tl.arange(0, NUM_PACKED_ROWS)[:, None] == i
            row_i_int32 = tl.sum(packed_row.to(tl.int32) * row_mask_2d, axis=0)  # (BLOCK_N,)
            row_i_int32 = row_i_int32  # shape (BLOCK_N,)
            for j in tl.static_range(GROUP_K):
                nibble = (row_i_int32 >> (4 * j)) & 0xF
                # Dequantize: (nibble - 8) * scale
                w_val = (nibble.to(tl.float32) - 8.0) * scales_row.to(tl.float32)
                w_val_f16 = w_val.to(tl.float16)  # (BLOCK_N,)
                # Row index in W_tile
                row_idx = i * GROUP_K + j
                row_mask = tl.arange(0, BLOCK_K)[:, None] == row_idx
                w_val_bcast = w_val_f16[None, :].broadcast_to((BLOCK_K, BLOCK_N))
                w_tile = tl.where(row_mask, w_val_bcast, w_tile)

        # ------- Accumulate: acc += A_tile @ W_tile ----------
        acc += tl.dot(a_tile, w_tile)

    # ------- Store result tile (C) ----------
    c_ptrs = C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(torch.float16))


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