import torch
import triton
import triton.language as tl

@triton.jit
def embedding_kernel(
    weight_ptr,       # f32 pointer, shape (V, D)
    idx_ptr,          # i64 pointer, shape (N,)
    out_ptr,          # f32 pointer, shape (N, D)
    N: tl.constexpr,  # number of rows to gather
    D: tl.constexpr,  # embedding dimension
):
    pid = tl.program_id(0)
    if pid >= N:
        return

    # Load the index for this output row
    idx = tl.load(idx_ptr + pid)                 # int64
    # Compute byte offset to the row in weight (weights are row-major contiguous)
    row_offset = idx * D                          # int64
    # Offset along the dimension
    offs = tl.arange(0, D)                       # int32
    # Load the entire row from weight
    weight_row = tl.load(weight_ptr + row_offset + offs)  # f32
    # Compute output row pointer and store
    out_row_ptr = out_ptr + pid * D
    tl.store(out_row_ptr + offs, weight_row)

def triton_run(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    assert weight.dtype == torch.float32
    assert idx.dtype == torch.int64
    assert weight.ndim == 2
    assert idx.ndim == 1
    V, D = weight.shape
    N = idx.shape[0]

    out = torch.empty(N, D, dtype=torch.float32, device=weight.device)

    grid = (N,)
    embedding_kernel[grid](
        weight, idx, out,
        N=N, D=D,
        num_warps=4,
        num_stages=2,
    )
    return out