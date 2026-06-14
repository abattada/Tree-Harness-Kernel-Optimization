import torch
import triton
import triton.language as tl


@triton.jit
def _embedding_kernel(weight_ptr, idx_ptr, out_ptr,
                      D: tl.constexpr, N: tl.constexpr,
                      BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    if row_id < N:
        # Load and cast the int64 index to int32 for cheaper address arithmetic.
        idx_val = tl.load(idx_ptr + row_id)
        idx_val32 = tl.cast(idx_val, tl.int32)
        # Iterate over the feature dimension in chunks.
        for d_start in tl.static_range(0, D, BLOCK_SIZE):
            d_offs = d_start + tl.arange(0, BLOCK_SIZE)
            # BLOCK_SIZE divides D exactly, so no mask is required.
            val = tl.load(weight_ptr + idx_val32 * D + d_offs)
            tl.store(out_ptr + row_id * D + d_offs, val)


def triton_run(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Embedding lookup: out[i, :] = weight[idx[i], :]
    weight: (V, D) float32
    idx:    (N,)   int64
    returns: (N, D) float32
    """
    V, D = weight.shape
    N = idx.shape[0]

    # Tuned BLOCK_SIZE: D=1024 is a power of two.  512 gives two
    # coalesced vectorizable chunks per row while keeping num_warps=16,
    # which balances occupancy and register pressure well on Blackwell.
    BLOCK_SIZE = 512
    assert D % BLOCK_SIZE == 0, f"Feature dimension {D} must be divisible by BLOCK_SIZE {BLOCK_SIZE}"

    out = torch.empty(N, D, dtype=weight.dtype, device=weight.device)
    grid = (N,)
    _embedding_kernel[grid](
        weight, idx, out, D, N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=BLOCK_SIZE // 32,
    )
    return out