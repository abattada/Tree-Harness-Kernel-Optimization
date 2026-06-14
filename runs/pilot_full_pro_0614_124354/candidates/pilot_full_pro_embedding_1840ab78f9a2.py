import torch
import triton
import triton.language as tl


@triton.jit
def _embedding_kernel(
    weight_ptr,
    idx_ptr,
    out_ptr,
    N,        # number of indices
    D,        # embedding dimension
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= N:
        return
    # Load the target row index for this output row
    row_idx = tl.load(idx_ptr + pid)        # int64 from i64 tensor
    # Base pointers for the weight row and output row
    weight_row = weight_ptr + row_idx * D
    out_row = out_ptr + pid * D

    # Each thread in the block handles a set of columns with stride BLOCK_SIZE.
    tid = tl.arange(0, BLOCK_SIZE)
    steps = D // BLOCK_SIZE                   # integer, 8 for D=1024, BLOCK_SIZE=128
    for step in range(steps):
        cols = tid + step * BLOCK_SIZE        # contiguous across threads
        data = tl.load(weight_row + cols)     # coalesced load
        tl.store(out_row + cols, data)        # coalesced store


def triton_run(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Gather rows from `weight` according to `idx`.
    Signature: triton_run(weight: f32[131072, 1024], idx: i64[1048576]) -> f32[1048576, 1024]
    """
    N = idx.numel()
    D = weight.size(1)
    assert weight.is_contiguous() and idx.is_contiguous()
    out = torch.empty((N, D), dtype=weight.dtype, device=weight.device)

    # Launch parameters tuned for high occupancy: small blocks, many programs
    BLOCK_SIZE = 128          # 4 warps
    num_warps = 4
    grid = (N,)

    _embedding_kernel[grid, num_warps](
        weight_ptr=weight,
        idx_ptr=idx,
        out_ptr=out,
        N=N,
        D=D,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out