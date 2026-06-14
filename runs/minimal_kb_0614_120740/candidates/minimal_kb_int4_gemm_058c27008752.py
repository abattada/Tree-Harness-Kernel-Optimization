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
    # Ensure K-block is a multiple of 8 (needed for nibble extraction)
    tl.static_assert(BLOCK_K % 8 == 0, "BLOCK_K must be a multiple of 8")
    tl.static_assert(BLOCK_K >= 8, "BLOCK_K must be at least 8")

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Number of nibble groups per K block
    NIBBLE_GROUPS = BLOCK_K // 8

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # ---- Load packed weight tile (int32) ----
        w_offs_row = offs_k // 8  # shape (BLOCK_K,)
        w = tl.load(
            w_ptr + w_offs_row[:, None] * stride_wk + offs_n[None, :] * stride_wn,
            mask=(w_offs_row[:, None] < (K // 8)) & (offs_n[None, :] < N),
            other=0,
        )
        # w now has shape (BLOCK_K, N) but note: each element in a row corresponds
        # to the same packed word for that row offset. Actually we loaded BLOCK_K
        # rows of w_packed? Wait: offs_k // 8 gives indices of w_packed for each
        # K index. So w has shape (BLOCK_K, BLOCK_N), which is redundant because
        # each consecutive 8 rows share the same packed word. It's okay but we
        # only need the first NIBBLE_GROUPS rows. We'll reshape later.
        # Better approach: load directly with shape (NIBBLE_GROUPS, BLOCK_N) using
        # offs_k[::8] // 8. Let's do:
        offs_k_nibble = k_start + tl.arange(0, NIBBLE_GROUPS) * 8
        w_offs_nibble = offs_k_nibble // 8  # (NIBBLE_GROUPS,)
        w_packed = tl.load(
            w_ptr + w_offs_nibble[:, None] * stride_wk + offs_n[None, :] * stride_wn,
            mask=(w_offs_nibble[:, None] < (K // 8)) & (offs_n[None, :] < N),
            other=0,
        )
        # w_packed shape: (NIBBLE_GROUPS, BLOCK_N)

        # ---- Load scales (per column, fp16) ----
        s = tl.load(scales_ptr + offs_n, mask=offs_n < N, other=0.0)

        # ---- For each nibble group, extract corresponding columns of a and dequantized weights ----
        for g in range(8):
            # Column indices in a for this group: k_start + g + (0, NIBBLE_GROUPS-1)*8
            k_offs = k_start + g + tl.arange(0, NIBBLE_GROUPS) * 8  # shape (NIBBLE_GROUPS,)
            a_slice = tl.load(
                a_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak,
                mask=(offs_m[:, None] < M) & (k_offs[None, :] < K),
                other=0.0,
            )  # shape (BLOCK_M, NIBBLE_GROUPS)

            # Extract nibble for group g from w_packed
            nibbles = ((w_packed >> (g * 4)) & 0xF).to(tl.float16)  # (NIBBLE_GROUPS, BLOCK_N)
            deq = (nibbles - 8.0) * s[None, :]  # (NIBBLE_GROUPS, BLOCK_N)

            # Accumulate
            acc += tl.dot(a_slice, deq)

    # ---- Store result ----
    c = acc.to(tl.float16)
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        c,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )

int4_gemm_kernel = triton.jit(int4_gemm_kernel)

# ---------------------------------------------------------------------------
# Public API: triton_run(a, w_packed, scales) -> output
# ---------------------------------------------------------------------------
def triton_run(a: torch.Tensor, w_packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    assert w_packed.shape[0] == K // 8
    N = w_packed.shape[1]
    assert scales.shape == (N,)

    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # Block sizes – all divide 4096 cleanly
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64  # must be multiple of 8

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