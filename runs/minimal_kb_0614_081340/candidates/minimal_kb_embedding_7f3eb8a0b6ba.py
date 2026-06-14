import torch
import triton
import triton.language as tl

@triton.jit
def embed_kernel(
    weight_ptr,        # [weight_rows, D] in row-major
    idx_ptr,           # [N] int64
    out_ptr,           # [N, D]
    N,                 # number of output rows
    D,                 # hidden dimension
    stride_weight_row, # = D (since contiguous)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    # Load the index for this output row
    idx = tl.load(idx_ptr + pid)
    # Compute base pointer to the row in weight
    weight_row_base = weight_ptr + idx * stride_weight_row
    # Load the entire row (D elements)
    offsets = tl.arange(0, BLOCK_SIZE)
    # Since D == BLOCK_SIZE and is power of 2, no mask needed
    row_data = tl.load(weight_row_base + offsets)
    # Store to output row
    out_row_base = out_ptr + pid * stride_weight_row
    tl.store(out_row_base + offsets, row_data)

def triton_run(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    weight: [131072, 1024] float32
    idx:    [1048576] int64
    returns: [1048576, 1024] float32
    """
    N, D = idx.shape[0], weight.shape[1]
    BLOCK_SIZE = D  # 1024
    out = torch.empty(N, D, device=weight.device, dtype=weight.dtype)
    grid = (N,)
    embed_kernel[grid](
        weight,
        idx,
        out,
        N,
        D,
        weight.stride(0),   # stride between rows (should be D)
        BLOCK_SIZE,
        num_warps=4,
        num_stages=4,
    )
    return out