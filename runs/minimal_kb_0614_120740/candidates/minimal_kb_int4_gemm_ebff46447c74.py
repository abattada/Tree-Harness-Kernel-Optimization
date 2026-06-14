import torch
import triton
import triton.language as tl

@triton.jit
def int4_matmul_kernel(
    a_ptr, w_packed_ptr, scales_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wpk, stride_wpn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Tile indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to A tile (M x K block)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k_start in range(0, K, BLOCK_K):
        k_remaining = K - k_start
        # Load A tile from global to shared memory
        a_tile = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining if k_remaining < BLOCK_K else None)

        # Load w_packed tile (K//8 x N block)
        # w_packed shape: (K//8, N)
        wp_k_start = k_start // 8
        offs_wpk = wp_k_start + tl.arange(0, BLOCK_K // 8)
        wp_ptrs = w_packed_ptr + offs_wpk[:, None] * stride_wpk + offs_n[None, :] * stride_wpn
        # Mask for the last tile in K (if BLOCK_K not multiple of 8? But K is multiple of 8)
        wp_tile = tl.load(wp_ptrs, mask=offs_wpk[:, None] < (K // 8))

        # Load scales tile (size N)
        scales_ptrs = scales_ptr + offs_n
        scales_tile = tl.load(scales_ptrs)

        # Dequantize w_packed tile into fp16 tile (BLOCK_K x BLOCK_N)
        # We'll compute directly in registers: for each of the 8 parts in the packed word
        w_tile = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float16)
        # Unroll over 8 groups
        for bit in range(8):
            # Shift and mask
            val = (wp_tile >> (bit * 4)) & 0xF
            val = val - 8  # int range [-8, 7]
            val = val.to(tl.float16) * scales_tile[None, :]
            # Place at row offset = bit
            w_tile = tl.where(tl.arange(0, BLOCK_K)[:, None] == bit, val, w_tile)

        # Dot product
        acc += tl.dot(a_tile.to(tl.float16), w_tile)

        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak

    # Store result
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16))


def triton_run(a: torch.Tensor, w_packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    N = scales.shape[0]
    assert w_packed.shape == (K // 8, N)
    assert a.dtype == torch.float16
    assert scales.dtype == torch.float16

    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # Tile sizes (tunable, pick multiples of 8 for K dimension)
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64  # must be multiple of 8

    grid = (M // BLOCK_M, N // BLOCK_N)
    int4_matmul_kernel[grid](
        a, w_packed, scales, c,
        M, N, K,
        a.stride(0), a.stride(1),
        w_packed.stride(0), w_packed.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
    )
    return c