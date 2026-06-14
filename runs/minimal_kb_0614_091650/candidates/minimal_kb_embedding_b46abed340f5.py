import torch
import triton
import triton.language as tl

@triton.jit
def embedding_kernel(
    weight_ptr, idx_ptr, out_ptr,
    idx_stride,          # stride of idx (always 1 for contiguous)
    weight_row_stride,   # stride from one row to next in weight (should be D)
    out_row_stride,      # stride from one row to next in out (should be D)
    D: tl.constexpr,     # embedding dimension (1024)
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    # load the index for this row
    idx = tl.load(idx_ptr + row * idx_stride, mask=row < tl.num_programs)
    # compute base pointer for the selected weight row
    weight_row_ptr = weight_ptr + idx * weight_row_stride
    out_row_ptr = out_ptr + row * out_row_stride
    # offsets along embedding dimension
    offs = tl.arange(0, BLOCK_D)
    # load the entire row (mask only needed if D not multiple of BLOCK_D, but here D == BLOCK_D)
    vals = tl.load(weight_row_ptr + offs, mask=offs < D, other=0.0)
    # store to output
    tl.store(out_row_ptr + offs, vals, mask=offs < D)

def triton_run(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    assert weight.dtype == torch.float32
    assert idx.dtype == torch.int64
    assert weight.shape[0] > 0  # at least one row
    D = weight.shape[1]
    n_rows = idx.shape[0]
    out = torch.empty(n_rows, D, dtype=torch.float32, device=weight.device)

    # Choose block size equal to embedding dimension for efficiency
    BLOCK_D = triton.next_power_of_2(D)  # 1024
    grid = (n_rows,)
    embedding_kernel[grid](
        weight, idx, out,
        idx.stride(0),
        weight.stride(0),
        out.stride(0),
        D,
        BLOCK_D,
        num_warps=4,
        num_stages=2,
    )
    return out