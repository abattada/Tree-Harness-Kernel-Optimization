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
    # Guard against out-of-range (though grid = N exactly)
    if pid < N:
        # Load index for this row
        index = tl.load(idx_ptr + pid)  # int64

        # Compute base pointer into weight: weight_ptr + index * D
        # Use tl.slow_ticks-free multiplication
        base_w = weight_ptr + index * stride_w

        # Compute base pointer into output: out_ptr + pid * stride_o
        base_o = out_ptr + pid * stride_o

        # Offsets along embedding dimension
        offsets = tl.arange(0, BLOCK_D)
        mask = offsets < D

        # Load and store
        val = tl.load(base_w + offsets, mask=mask)
        tl.store(base_o + offsets, val, mask=mask)


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

    out = torch.empty(N, D, dtype=torch.float32, device=weight.device)

    # Block size: load entire row at once (D)
    BLOCK_D = D  # 1024

    grid = (N,)
    embedding_kernel[grid](
        weight, idx, out,
        N, D,
        weight.stride(0), out.stride(0),
        BLOCK_D,
        num_warps=4,
        num_stages=2,
    )
    return out