import torch
import triton
import triton.language as tl

def _cdiv(x, y):
    return (x + y - 1) // y

@triton.jit
def embedding_kernel(
    weight_ptr,          # *fp32: [E, D]
    idx_ptr,             # *i64:   [N]
    out_ptr,             # *fp32: [N, D]
    N,                   # number of output rows
    D: tl.constexpr,     # embedding dimension (1024)
    BLOCK_ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    rows_start = pid * BLOCK_ROWS
    offs = rows_start + tl.arange(0, BLOCK_ROWS)
    mask = offs < N

    # Load indices for this block of rows
    idx_vals = tl.load(idx_ptr + offs, mask=mask, other=0)

    # Process each row in the block
    for i in range(BLOCK_ROWS):
        if mask[i]:
            rid = idx_vals[i]
            # Load a whole row of the weight matrix
            row = tl.load(weight_ptr + rid * D + tl.arange(0, D),
                          mask=None, other=0.0)
            # Store to output
            tl.store(out_ptr + (rows_start + i) * D + tl.arange(0, D), row)


def triton_run(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gathered embedding: out[i] = weight[idx[i]] for all i.
    weight: [131072, 1024] f32
    idx:    [1048576] i64
    returns: [1048576, 1024] f32
    """
    assert weight.is_cuda and idx.is_cuda, "Tensors must be on CUDA"
    assert weight.dtype == torch.float32
    assert idx.dtype == torch.int64
    assert weight.ndim == 2 and idx.ndim == 1
    E, D = weight.shape
    N = idx.numel()
    assert D == 1024, "D must be 1024"
    assert weight.shape[0] == 131072 and N == 1048576, "Unexpected shapes"

    # Ensure idx is contiguous for efficient loading
    idx = idx.contiguous()

    out = torch.empty(N, D, dtype=weight.dtype, device=weight.device)

    # Block size: process multiple rows per program
    BLOCK_ROWS = 256
    grid = (_cdiv(N, BLOCK_ROWS),)

    embedding_kernel[grid](
        weight,
        idx,
        out,
        N,
        D=D,
        BLOCK_ROWS=BLOCK_ROWS,
        num_warps=4,
        num_stages=2,
    )
    return out