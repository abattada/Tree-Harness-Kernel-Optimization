import torch
import triton
import triton.language as tl

@triton.jit
def embedding_persistent_kernel(
    weight_ptr, idx_ptr, out_ptr,
    N, D,
    stride_weight_0, stride_weight_1,
    stride_out_0, stride_out_1,
    BLOCK_D: tl.constexpr,
):
    # Persistent kernel: each program processes a chunk of rows in a grid-stride loop
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    # Compute the starting row for this program
    row = pid
    # Loop over rows assigned to this program
    while row < N:
        # Load the index for this row
        idx = tl.load(idx_ptr + row)

        # Compute base pointer for the weight row
        weight_row_ptr = weight_ptr + idx * stride_weight_0

        # Offsets along the embedding dimension
        offs = tl.arange(0, BLOCK_D)
        mask = offs < D

        # Load the weight row (evict_first because it is read only once)
        w = tl.load(weight_row_ptr + offs * stride_weight_1, mask=mask, other=0.0,
                    eviction_policy='evict_first')

        # Store to output row
        out_row_ptr = out_ptr + row * stride_out_0
        tl.store(out_row_ptr + offs * stride_out_1, w, mask=mask)

        # Move to next row assigned to this program
        row += num_programs


def triton_run(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    assert weight.dtype == torch.float32
    assert idx.dtype == torch.int64
    assert weight.is_contiguous()
    vocab, D = weight.shape
    N = idx.shape[0]

    out = torch.empty(N, D, dtype=torch.float32, device=weight.device)

    # Launch a moderate number of programs (persistent kernel)
    # On RTX 5090, we use enough programs to keep all SMs busy
    # 1024 programs is a good trade-off between occupancy and launch overhead
    num_programs = min(N, 1024)

    grid = (num_programs,)
    embedding_persistent_kernel[grid](
        weight, idx, out,
        N, D,
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_D=D,
        num_warps=4,
        num_stages=2,
    )

    return out