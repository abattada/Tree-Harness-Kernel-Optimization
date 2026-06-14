import torch
import triton
import triton.language as tl

@triton.jit
def embedding_kernel_2d(
    weight_ptr, idx_ptr, out_ptr,
    N, D,
    stride_weight_0, stride_weight_1,
    stride_out_0, stride_out_1,
    BLOCK_D: tl.constexpr,
):
    # Program ID decomposition
    pid_row = tl.program_id(0)
    pid_d   = tl.program_id(1)
    if pid_row >= N:
        return

    # Load index for this row
    idx = tl.load(idx_ptr + pid_row, eviction_policy='evict_first')

    # Offsets along embedding dimension for this block
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    # Base pointer to weight row
    weight_row_base = weight_ptr + idx * stride_weight_0

    # Load weight elements
    w = tl.load(
        weight_row_base + offs_d * stride_weight_1,
        mask=mask_d,
        other=0.0,
        eviction_policy='evict_first'
    )

    # Base pointer to output row
    out_row_base = out_ptr + pid_row * stride_out_0

    # Store output elements
    tl.store(
        out_row_base + offs_d * stride_out_1,
        w,
        mask=mask_d
    )


def triton_run(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    assert weight.dtype == torch.float32
    assert idx.dtype == torch.int64
    assert weight.is_contiguous()
    vocab, D = weight.shape
    N = idx.shape[0]

    out = torch.empty(N, D, dtype=torch.float32, device=weight.device)

    # Tuned block size for higher occupancy: 256 elements per program
    BLOCK_D = 256
    num_d_blocks = (D + BLOCK_D - 1) // BLOCK_D  # ceil division

    grid = (N, num_d_blocks)

    embedding_kernel_2d[grid](
        weight, idx, out,
        N, D,
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_D=BLOCK_D,
        num_warps=2,
        num_stages=1,
    )

    return out