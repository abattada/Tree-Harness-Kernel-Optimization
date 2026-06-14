import torch
import triton
import triton.language as tl

@triton.jit
def embedding_kernel(
    weight_ptr, idx_ptr, out_ptr,
    N, D,
    stride_weight_0, stride_weight_1,
    stride_out_0, stride_out_1,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)
    row = pid
    # Grid-stride loop: each program processes multiple rows
    while row < N:
        # Load index for this row
        idx = tl.load(idx_ptr + row)
        # Compute base pointer for the weight row
        weight_row_base = weight_ptr + idx * stride_weight_0
        # Offsets along the embedding dimension
        offsets = tl.arange(0, BLOCK_D)
        mask = offsets < D
        # Load weight row; use evict_first since each row is used only once
        w = tl.load(
            weight_row_base + offsets * stride_weight_1,
            mask=mask,
            other=0.0,
            eviction_policy='evict_first',
        )
        # Compute base pointer for output row
        out_row_base = out_ptr + row * stride_out_0
        # Store output row
        tl.store(
            out_row_base + offsets * stride_out_1,
            w,
            mask=mask,
            eviction_policy='evict_first',
        )
        # Move to next row for this program
        row += grid_size


def triton_run(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    # Assert types and contiguity
    assert weight.dtype == torch.float32
    assert idx.dtype == torch.int64
    assert weight.is_contiguous()
    assert idx.is_contiguous()
    vocab, D = weight.shape
    N = idx.shape[0]
    out = torch.empty(N, D, dtype=torch.float32, device=weight.device)

    # Launch fewer programs to amortize overhead; each program processes ~N/grid rows
    # 1024 is a reasonable tradeoff for a 170-SM GPU
    grid = (1024,)
    embedding_kernel[grid](
        weight, idx, out,
        N, D,
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_D=D,
        num_warps=4,
        num_stages=2,
    )

    return out