import torch
import triton
import triton.language as tl


@triton.jit
def _embedding_kernel(weight_ptr, idx_ptr, out_ptr,
                      D: tl.constexpr, N: tl.constexpr,
                      BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    if row_id < N:
        # Load the (int64) index and cast to int32 for cheaper address arithmetic.
        idx_val = tl.load(idx_ptr + row_id)
        idx_val32 = tl.cast(idx_val, tl.int32)
        # Iterate over the feature dimension in chunks.
        for d_start in tl.static_range(0, D, BLOCK_SIZE):
            d_offs = d_start + tl.arange(0, BLOCK_SIZE)
            # Mask is only needed if D is not a multiple of BLOCK_SIZE;
            # we keep it for robustness, the compiler may optimize it away.
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
    
    # Tuned block size: larger blocks reduce loop overhead and allow more warps.
    BLOCK_SIZE = 512
    assert D % BLOCK_SIZE == 0, f"D={D} must be a multiple of BLOCK_SIZE={BLOCK_SIZE}"
    
    out = torch.empty(N, D, dtype=weight.dtype, device=weight.device)
    grid = (N,)
    _embedding_kernel[grid](
        weight, idx, out, D, N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=BLOCK_SIZE // 32,   # 16 warps for 512 threads
        num_stages=4,                 # pipelining across loop iterations
    )
    return out