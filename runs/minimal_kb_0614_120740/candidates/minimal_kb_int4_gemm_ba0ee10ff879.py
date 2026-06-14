import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel for int4 GEMM:  a @ dequant(w_packed, scales)
# ---------------------------------------------------------------------------
def int4_gemm_kernel(
    a_ptr, w_ptr, scales_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wk, stride_wn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row/column offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator (float32 for precision)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # ---- Load A tile (fp16) ----
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )

        # ---- Load packed weight tile (int32) ----
        # k_vals_8: index into packed weight rows (each packed word holds 8 int4 values)
        # k_vals_mod: nibble position within the packed word
        k_vals = tl.arange(0, BLOCK_K)
        k_vals_8 = k_vals // 8
        k_vals_mod = k_vals % 8

        w_word = tl.load(
            w_ptr + k_vals_8[:, None] * stride_wk + offs_n[None, :] * stride_wn,
            mask=(k_vals_8[:, None] < (K // 8)) & (offs_n[None, :] < N),
            other=0,
        )

        # ---- Load scales (per column, fp16) ----
        s = tl.load(scales_ptr + offs_n, mask=offs_n < N, other=0.0)

        # ---- Dequantize: extract nibbles and form (BLOCK_K, BLOCK_N) fp16 ----
        nibble = (w_word >> (k_vals_mod[:, None] * 4)) & 0xF
        w_fp = (nibble.to(tl.float16) - 8.0) * s[None, :]

        # ---- Accumulate dot product (tensor core) ----
        acc += tl.dot(a, w_fp)

    # ---- Store result ----
    c = acc.to(tl.float16)
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        c,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )

# Wrap with triton.jit (functional form to avoid @ decorator)
int4_gemm_kernel = triton.jit(int4_gemm_kernel)

# ---------------------------------------------------------------------------
# Public API: triton_run(a, w_packed, scales) -> output
# ---------------------------------------------------------------------------
def triton_run(a: torch.Tensor, w_packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    assert w_packed.shape[0] == K // 8
    N = w_packed.shape[1]
    assert scales.shape == (N,)

    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # Block sizes – all divide 4096 cleanly
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    grid = (M // BLOCK_M, N // BLOCK_N)

    int4_gemm_kernel[grid](
        a, w_packed, scales, c,
        M, N, K,
        a.stride(0), a.stride(1),
        w_packed.stride(0), w_packed.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
        num_warps=4,
        num_stages=4,
    )

    return c