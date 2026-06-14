import torch
import triton
import triton.language as tl


@triton.jit
def _embedding_kernel(weight_ptr, idx_ptr, out_ptr,
                      D: tl.constexpr,
                      N: tl.constexpr,
                      BLOCK_ROWS: tl.constexpr,
                      BLOCK_D: tl.constexpr):
    """
    Gather rows: out[i, :] = weight[idx[i], :]
    Each program processes BLOCK_ROWS rows; each row is processed by a
    1‑D thread block of size BLOCK_D (one thread per feature element).
    """
    pid = tl.program_id(0)
    row_start = pid * BLOCK_ROWS

    for r in tl.static_range(0, BLOCK_ROWS):
        row_id = row_start + r
        if row_id < N:
            # load the index and cast to int32 for cheaper address arithmetic
            idx_val = tl.load(idx_ptr + row_id)
            idx_val32 = tl.cast(idx_val, tl.int32)

            # iterate over the feature dimension with the chosen tile size
            for d_start in tl.static_range(0, D, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                # mask is only active if D is not a multiple of BLOCK_D;
                # we keep it for safety, the compiler will eliminate it if unneeded.
                mask = d_offs < D
                val = tl.load(weight_ptr + idx_val32 * D + d_offs, mask=mask)
                tl.store(out_ptr + row_id * D + d_offs, val, mask=mask)


def triton_run(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Embedding lookup: out[i, :] = weight[idx[i], :]
    weight: (V, D) float32
    idx:    (N,)   int64
    returns: (N, D) float32
    """
    V, D = weight.shape
    N = idx.shape[0]

    # Tuned tile sizes for the target shape (D=1024, N≈1M).
    # BLOCK_D = 1024 uses the whole feature row in one shot (1 warp per 32 threads).
    # BLOCK_ROWS = 128 gives a grid of ~8k programs, reducing launch overhead.
    BLOCK_D = 1024
    BLOCK_ROWS = 128
    assert D % BLOCK_D == 0, f"D={D} must be a multiple of BLOCK_D={BLOCK_D}"

    out = torch.empty(N, D, dtype=weight.dtype, device=weight.device)

    # 1‑D grid along rows; each program handles BLOCK_ROWS rows.
    grid = (triton.cdiv(N, BLOCK_ROWS),)

    _embedding_kernel[grid](
        weight, idx, out,
        D=D,
        N=N,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_D=BLOCK_D,
        num_warps=BLOCK_D // 32,   # 32 warps -> 1024 threads
    )
    return out