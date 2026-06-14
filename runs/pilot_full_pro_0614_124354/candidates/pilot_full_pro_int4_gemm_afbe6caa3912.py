import torch
import triton
import triton.language as tl

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
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # scales are constant along K, load once in fp32 for the multiplication
    scales_ptrs = scales_ptr + rn * stride_s
    scales_row = tl.load(scales_ptrs).to(tl.float32)  # (BLOCK_N,)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        for i in tl.static_range(0, BLOCK_K, GROUP_K):
            # --- load a (BLOCK_M, 8) tile of A ---
            a_cols = k_start + i + tl.arange(0, GROUP_K)
            a_ptrs = A_ptr + rm[:, None] * stride_am + a_cols[None, :] * stride_ak
            a_chunk = tl.load(a_ptrs)  # fp16

            # --- load one packed int32 row (8 weight values per column) ---
            w_row_offset = (k_start + i) // GROUP_K
            w_ptrs = W_packed_ptr + w_row_offset * stride_wk + rn * stride_wn
            packed_row = tl.load(w_ptrs)  # int32, shape (BLOCK_N,)

            # --- dequantize: extract nibbles, subtract 8, multiply by scales ---
            shift = tl.arange(0, 32, 4)            # (8,)  = 0,4,8,...,28
            nibbles = (packed_row[None, :] >> shift[:, None]) & 0xF   # (8, BLOCK_N)
            w_sub_f32 = (nibbles.to(tl.float32) - 8.0) * scales_row[None, :]
            w_sub_f16 = w_sub_f32.to(tl.float16)   # (8, BLOCK_N)

            # --- accumulate partial product ---
            acc += tl.dot(a_chunk, w_sub_f16)

    # store the result (fp16)
    c_ptrs = C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16))


def triton_run(a: torch.Tensor, w_packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    _, N = w_packed.shape
    assert K % 8 == 0, "K must be divisible by 8"

    C = torch.empty((M, N), dtype=a.dtype, device=a.device)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _int4_gemm_kernel[grid](
        a, w_packed, scales, C,
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