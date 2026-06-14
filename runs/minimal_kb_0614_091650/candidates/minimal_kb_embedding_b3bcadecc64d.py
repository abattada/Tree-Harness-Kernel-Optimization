import torch
import triton
import triton.language as tl

@triton.jit
def embedding_kernel(
    weight_ptr, idx_ptr, out_ptr,
    N: tl.constexpr, D: tl.constexpr,
    stride_w: tl.constexpr,  # weight row stride (always D)
    stride_o: tl.constexpr,  # output row stride (always D)
    BLOCK_D: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    """
    Persistent embedding gather.
    Each program processes ROWS_PER_PROG contiguous rows, looping over them.
    """
    pid = tl.program_id(0)
    start = pid * ROWS_PER_PROG
    end = tl.minimum(start + ROWS_PER_PROG, N)

    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    i = start
    while i < end:
        # Load index for this row
        idx = tl.load(idx_ptr + i, eviction_policy='evict_first')

        # Base pointers for weight and output
        base_w = weight_ptr + idx * stride_w
        base_o = out_ptr + i * stride_o

        # Load and store the embedding vector
        val = tl.load(base_w + offs, mask=mask, eviction_policy='evict_first')
        tl.store(base_o + offs, val, mask=mask)

        i += 1


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

    BLOCK_D = D  # 1024, trivially divides D
    ROWS_PER_PROG = 512  # constant, chosen for good utilization

    # Grid size: split rows into chunks of ROWS_PER_PROG
    grid = (triton.cdiv(N, ROWS_PER_PROG),)

    embedding_kernel[grid](
        weight, idx, out,
        N, D,
        weight.stride(0), out.stride(0),
        BLOCK_D, ROWS_PER_PROG,
        num_warps=8,
        num_stages=2,
    )
    return out