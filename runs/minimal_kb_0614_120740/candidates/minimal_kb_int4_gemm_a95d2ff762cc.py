import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel: fused int4 dequantize + matrix multiply
# ---------------------------------------------------------------------------
@triton.jit
def int4_gemm_kernel(
    a_ptr, w_packed_ptr, scales_ptr, c_ptr,
    M, N, K,
    stride_a_m, stride_a_k,
    stride_wp_pk, stride_wp_n,
    stride_c_m, stride_c_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator (f32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Pre-load scales for this N tile (they are reused across K)
    scales_tile = tl.load(scales_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float16)

    # Main K loop
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load A tile (f16)  [BLOCK_M, BLOCK_K]
        a_tile = tl.load(
            a_ptr + offs_m[:, None] * stride_a_m + offs_k[None, :] * stride_a_k,
            mask=offs_k[None, :] < K,
            other=0.0,
        )

        # Load packed weight tile (int32) [BLOCK_K//8, BLOCK_N]
        offs_pk = tl.arange(0, BLOCK_K // 8)
        pk_start = k_start // 8
        w_packed_tile = tl.load(
            w_packed_ptr + (pk_start + offs_pk)[:, None] * stride_wp_pk + offs_n[None, :] * stride_wp_n,
            mask=(offs_n[None, :] < N) & ((pk_start + offs_pk)[:, None] < (K // 8)),
            other=0,
        )

        # Dequantize packed weights to f16 weight tile [BLOCK_K, BLOCK_N]
        weight_tile = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float16)
        for pk_rel in tl.static_range(BLOCK_K // 8):
            w_packed_row = w_packed_tile[pk_rel, :]          # shape [BLOCK_N]
            for j in tl.static_range(8):
                nibble = (w_packed_row >> (j * 4)) & 0xF
                nibble_f16 = nibble.to(tl.float16)
                # Dequantize: (value - 8) * scale
                weight_tile[pk_rel * 8 + j, :] = (nibble_f16 - 8.0) * scales_tile

        # Tensor core matmul with f32 accumulation
        acc = tl.dot(a_tile, weight_tile, acc)

    # Store result (f16)
    c_tile = acc.to(tl.float16)
    mask_m = offs_m[:, None] < M
    mask_n = offs_n[None, :] < N
    tl.store(
        c_ptr + offs_m[:, None] * stride_c_m + offs_n[None, :] * stride_c_n,
        c_tile,
        mask=mask_m & mask_n,
    )


# ---------------------------------------------------------------------------
# Public entry point (must be named triton_run)
# ---------------------------------------------------------------------------
def triton_run(a: torch.Tensor,
               w_packed: torch.Tensor,
               scales: torch.Tensor) -> torch.Tensor:
    """
    a       : f16 [M, K]
    w_packed: i32 [K//8, N]
    scales  : f16 [N]
    returns : f16 [M, N] = a @ dequant(w_packed, scales)
    """
    M, K = a.shape
    N = w_packed.shape[1]
    assert w_packed.shape[0] == K // 8
    assert scales.shape[0] == N

    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # Tuned for RTX 5090 (Blackwell): block sizes that divide 4096 evenly
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32   # must be multiple of 8 (packing factor)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    int4_gemm_kernel[grid](
        a, w_packed, scales, c,
        M, N, K,
        a.stride(0), a.stride(1),
        w_packed.stride(0), w_packed.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c