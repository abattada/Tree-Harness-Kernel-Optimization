import torch
import triton
import triton.language as tl

@triton.jit
def embedding_kernel(
    weight_ptr, idx_ptr, out_ptr,
    N: tl.constexpr, D: tl.constexpr,
    stride_w: tl.constexpr,  # weight stride for rows (should be D)
    stride_o: tl.constexpr,  # output stride for rows (should be D)
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)
    row = pid
    # Grid-stride loop over rows
    while row < N:
        # Load index for this row
        idx = tl.load(idx_ptr + row)  # int64

        # Base pointers
        base_w = weight_ptr + idx * stride_w
        base_o = out_ptr + row * stride_o

        # Offsets along the embedding dimension (0..D-1)
        offsets = tl.arange(0, BLOCK_D)  # already D elements

        # No mask needed because D == BLOCK_D
        val = tl.load(base_w + offsets, eviction_policy='evict_first')
        tl.store(base_o + offsets, val, eviction_policy='evict_first')

        row += grid_size


def triton_run(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Embedding lookup: out[i] = weight[idx[i]]
    weight: (vocab_size, D) float32
    idx: (N,) int64
    Returns: (N, D) float32
    """
    N = idx.shape[0]
    D = weight.shape[1]
    assert weight.dtype == torch.float32
    assert idx.dtype == torch.long
    assert D == 1024  # fixed for this kernel

    out = torch.empty(N, D, dtype=torch.float32, device=weight.device)

    # Use a moderate grid size – fewer programs to reduce launch overhead,
    # each loops over multiple rows.
    grid_size = min(N, 512)  # 512 programs
    BLOCK_D = D  # 1024

    grid = (grid_size,)
    embedding_kernel[grid](
        weight, idx, out,
        N, D,
        weight.stride(0), out.stride(0),
        BLOCK_D,
        num_warps=4,
        num_stages=2,
    )
    return out