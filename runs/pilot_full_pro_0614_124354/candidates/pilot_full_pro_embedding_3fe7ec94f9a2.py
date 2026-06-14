import torch
import triton
import triton.language as tl


@triton.jit
def _embedding_kernel(weight_ptr, idx_ptr, out_ptr,
                      D: tl.constexpr, N: tl.constexpr,
                      BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    # Process only if row_id is within range (grid covers N exactly)
    if row_id < N:
        # Load the (int64) index and cast to int32 for cheaper address arithmetic
        idx_val = tl.load(idx_ptr + row_id)
        idx_val32 = tl.cast(idx_val, tl.int32)
        # Base offsets for the row
        w_base = idx_val32 * D
        o_base = row_id * D
        # Iterate over the feature dimension in chunks.
        # Because D is a multiple of BLOCK_SIZE, we can drop the mask
        # and tell the compiler the addresses are contiguous and aligned.
        for d_start in tl.static_range(0, D, BLOCK_SIZE):
            d_offs = d_start + tl.arange(0, BLOCK_SIZE)
            # tl.multiple_of hints improve vectorization
            val = tl.load(weight_ptr + w_base + d_offs,
                          mask=None,
                          cache_modifier='.cg',           # cache-global (streaming)
                          eviction_policy='evict_first')   # don't pollute L2 with weight
            tl.store(out_ptr + o_base + d_offs, val)


def triton_run(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Embedding lookup: out[i, :] = weight[idx[i], :]
    weight: (V, D) float32
    idx:    (N,)   int64
    returns: (N, D) float32
    """
    V, D = weight.shape
    N = idx.shape[0]

    # Tuned block size.  Must divide D.
    BLOCK_SIZE = 256
    assert D % BLOCK_SIZE == 0, f"D={D} must be a multiple of BLOCK_SIZE={BLOCK_SIZE}"

    out = torch.empty(N, D, dtype=weight.dtype, device=weight.device)
    grid = (N,)

    # Tuned launch configuration:
    #   BLOCK_SIZE = 256  gives 256 threads -> 8 warps (256/32 = 8)
    #   num_warps = 8  (the minimum for this thread count)
    #   num_stages = 4 – pipelining the two loads (index + weight) and store
    _embedding_kernel[grid](
        weight, idx, out,
        D, N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=4,
    )
    return out