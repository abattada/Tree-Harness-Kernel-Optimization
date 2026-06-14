import torch
import triton
import triton.language as tl

@triton.jit
def addmm_kernel(
    bias_ptr,
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program IDs: pid_m along rows, pid_n along columns
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Masks (kept for safety even though dimensions divide the tile sizes)
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Load the bias for the rows handled by this block (promote to fp32)
    bias_vals = tl.load(bias_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main K-loop
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Tile from A: shape (BLOCK_M, BLOCK_K)
        a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
        a_tile = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Tile from B: shape (BLOCK_K, BLOCK_N)
        b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]
        b_tile = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Tensor-core matmul: fp16 in, fp32 out
        acc += tl.dot(a_tile, b_tile, out_dtype=tl.float32)

    # Add bias (broadcast along columns)
    acc += bias_vals[:, None]

    # Store tile in fp16
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def triton_run(bias: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    N = b.shape[1]
    assert bias.shape == (M,), "bias must have shape (M,)"

    c = torch.empty((M, N), dtype=torch.float16, device=a.device)

    # Seed tuning knobs – easily adjustable later
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    num_warps = 8
    num_stages = 4

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    addmm_kernel[grid](
        bias, a, b, c,
        M, N, K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return c