import torch
import triton
import triton.language as tl

@triton.jit
def embed_kernel(weight_ptr, idx_ptr, out_ptr,
                 N, D: tl.constexpr, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0)
    row_offs = pid * D

    # Load the index for this row
    idx = tl.load(idx_ptr + pid)

    # Compute base pointer into the weight matrix
    row_start = weight_ptr + idx * D

    # Vectorized load of the entire row
    offs = tl.arange(0, BLOCK_D)
    x = tl.load(row_start + offs)

    # Store to output
    tl.store(out_ptr + row_offs + offs, x)

def triton_run(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    weight: float32 [vocab_size, dim]  (131072 x 1024)
    idx:    int64  [batch]            (1048576)
    returns: float32 [batch, dim]
    """
    assert weight.is_cuda and idx.is_cuda
    vocab, D = weight.shape
    N = idx.shape[0]  # number of rows to gather

    out = torch.empty(N, D, device='cuda', dtype=torch.float32)

    # Launch one program per output row; BLOCK_D = 1024 (the full dim)
    grid = (N,)
    embed_kernel[grid](
        weight, idx, out,
        N, D, BLOCK_D=1024,
        num_warps=4,  # conservative
    )

    return out