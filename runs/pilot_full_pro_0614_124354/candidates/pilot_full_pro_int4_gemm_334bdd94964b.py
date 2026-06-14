import torch
import triton
import triton.language as tl


# Helper constant: the factor 8 for packed int4 layout.
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
    Computes C = A @ W, where W is dequantized on the fly from packed int4.
    Each program computes a BLOCK_M x BLOCK_N tile of C.
    The K dimension is iterated in blocks of BLOCK_K, internally split into
    micro-steps of 8 to unpack the 4-bit weights.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row/column indices for this block (with optional boundary mask).
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm_mask = rm < M
    rn_mask = rn < N

    # Load the scaling factors for the N columns of this block.
    # These are constant for the whole K loop.
    scales_ptrs = scales_ptr + rn * stride_s
    scales_row = tl.load(scales_ptrs, mask=rn_mask, other=0.0)  # fp16

    # Accumulator tile in fp32.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over the K dimension in steps of BLOCK_K.
    for k_start in range(0, K, BLOCK_K):
        # For each K block, we further split into groups of 8 to unpack int4.
        for k_micro in range(0, BLOCK_K, GROUP_K):
            k = k_start + k_micro
            k_valid = k + tl.arange(0, GROUP_K)  # 8 consecutive K indices

            # ---- Load A tile (BLOCK_M, 8) for this K micro-step ----
            a_ptrs = (A_ptr
                      + rm[:, None] * stride_am
                      + k_valid[None, :] * stride_ak)
            a_tile = tl.load(
                a_ptrs,
                mask=(rm_mask[:, None] & (k_valid[None, :] < K)),
                other=0.0,
            )  # shape (BLOCK_M, GROUP_K), dtype fp16

            # ---- Load one row of packed weights for k // 8 ----
            w_row_idx = k // GROUP_K
            w_row_ptrs = W_packed_ptr + w_row_idx * stride_wk + rn * stride_wn
            w_packed_row = tl.load(w_row_ptrs, mask=rn_mask, other=0)  # int32

            # ---- Unpack the int32 row into 8 rows of fp16 weights ----
            # w_tile will be (GROUP_K, BLOCK_N)
            w_tile = tl.zeros((GROUP_K, BLOCK_N), dtype=tl.float16)
            for j in tl.static_range(GROUP_K):
                # Extract the j-th nibble.
                nibble = (w_packed_row >> (4 * j)) & 0xF
                # Dequantize: (nibble - 8) * scale
                w_val_fp32 = (nibble.to(tl.float32) - 8.0) * scales_row.to(tl.float32)
                w_val = w_val_fp32.to(tl.float16)  # shape (BLOCK_N,)
                # Place into the j-th row of w_tile.
                # Create a mask that selects row j.
                row_mask = tl.arange(0, GROUP_K)[:, None] == j
                w_val_bcast = w_val[None, :].broadcast_to((GROUP_K, BLOCK_N))
                w_tile = tl.where(row_mask, w_val_bcast, w_tile)

            # ---- Accumulate matmul for this micro-step ----
            acc += tl.dot(a_tile, w_tile)  # (BLOCK_M, BLOCK_N)

    # ---- Store the result tile ----
    c_ptrs = C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=(rm_mask[:, None] & rn_mask[None, :]))


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

    # Tile sizes: powers of two that divide the matrix dimensions evenly.
    # These are constants chosen for a good baseline on Blackwell.
    BLOCK_M = 64
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
        num_warps=4,
        num_stages=2,
    )

    return C