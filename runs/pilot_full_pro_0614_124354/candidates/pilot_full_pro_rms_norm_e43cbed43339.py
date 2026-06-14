import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=32),
    ],
    key=[],
)
@triton.jit
def rms_norm_kernel(
    x_ptr,
    y_ptr,
    N,
    D: tl.constexpr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    RMS normalization without affine weights (shared‑memory fusion).
    Each block processes one row: loads the whole row into shared memory,
    computes sum of squares, reduces intra‑block, then scales and stores
    from shared memory – reading global memory only once.
    """
    tl.static_assert(D % BLOCK_SIZE == 0, "D must be divisible by BLOCK_SIZE")

    row_id = tl.program_id(0)
    if row_id >= N:
        return

    row_offset = row_id * D

    # Shared memory: row cache + per‑thread partial sums
    row_data = tl.static_shared_memory(shape=(D,), dtype=tl.float32)
    partial_sums = tl.static_shared_memory(shape=(BLOCK_SIZE,), dtype=tl.float32)

    tid = tl.arange(0, BLOCK_SIZE)

    # ---------- Pass 1: load row, compute per‑thread sum of squares ----------
    psum_sq = 0.0
    for start in range(0, D, BLOCK_SIZE):
        idx = start + tid
        mask = idx < D
        val = tl.load(x_ptr + row_offset + idx, mask=mask, other=0.0)
        tl.store(row_data + idx, val, mask=mask)
        psum_sq += val * val

    # Store per‑thread partial sums
    tl.store(partial_sums + tid, psum_sq)

    tl.debug_barrier()

    # ---------- Pass 2: intra‑block reduction ----------
    stride = BLOCK_SIZE // 2
    while stride > 0:
        pid = tid
        if pid < stride:
            a = tl.load(partial_sums + pid)
            b = tl.load(partial_sums + pid + stride)
            tl.store(partial_sums + pid, a + b)
        stride //= 2
        tl.debug_barrier()

    # All threads read the final total sum (resides at partial_sums[0])
    sum_sq = tl.load(partial_sums + 0)
    mean_sq = sum_sq / D
    rms = tl.rsqrt(mean_sq + eps)

    # ---------- Pass 3: normalize and store ----------
    for start in range(0, D, BLOCK_SIZE):
        idx = start + tid
        mask = idx < D
        val = tl.load(row_data + idx, mask=mask)
        out = val * rms
        tl.store(y_ptr + row_offset + idx, out, mask=mask)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    Applies RMS normalization to the last dimension.
    Signature: triton_run(x: f32[8192, 4096]) -> f32[8192, 4096]
    """
    N, D = x.shape
    assert D == 4096, f"Expected last dim 4096, got {D}"
    x = x.contiguous()
    y = torch.empty_like(x)

    grid = (N,)
    # D and eps are passed as keyword arguments to enable constexpr conversion
    rms_norm_kernel[grid](x, y, N, D=D, eps=1e-5)
    return y