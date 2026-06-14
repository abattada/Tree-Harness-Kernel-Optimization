import torch
import triton
import triton.language as tl


@triton.jit
def _embedding_kernel(
    weight_ptr,
    idx_ptr,
    out_ptr,
    N,                          # total output rows = 1048576
    D,                          # embedding dimension = 1024 (not used at runtime, kept for clarity)
    stride_weight_0: tl.constexpr,   # row stride of weight
    stride_weight_1: tl.constexpr,   # element stride within a row
    stride_out_0: tl.constexpr,     # row stride of output
    stride_out_1: tl.constexpr,     # element stride within a row
    BLOCK_D: tl.constexpr,         # = 1024, exactly equals D
):
    pid = tl.program_id(0)
    if pid >= N:
        return

    # Load the index for this output row
    idx_val = tl.load(idx_ptr + pid)

    # Base pointer to the weight row
    weight_row_base = weight_ptr + idx_val * stride_weight_0

    # Offsets along the embedding dimension – no mask needed because BLOCK_D == D
    offs = tl.arange(0, BLOCK_D)

    # Gather the row (coalesced load)
    w = tl.load(weight_row_base + offs * stride_weight_1)

    # Base pointer to the output row
    out_row_base = out_ptr + pid * stride_out_0
    tl.store(out_row_base + offs * stride_out_1, w)


def triton_run(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Signature: triton_run(weight: f32[131072, 1024], idx: i64[1048576])
                -> f32[1048576, 1024]
    """
    assert weight.dtype == torch.float32
    assert idx.dtype == torch.int64
    assert weight.is_contiguous()

    D = weight.shape[1]
    assert D == 1024, "Embedding dimension must be 1024 as per signature"
    N = idx.numel()

    # Allocate contiguous output tensor
    out = torch.empty(N, D, dtype=weight.dtype, device=weight.device)

    # One program per output row – full GPU occupancy for large N
    grid = (N,)

    _embedding_kernel[grid](
        weight,
        idx,
        out,
        N,
        D,
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_D=D,
        num_warps=4,
        num_stages=2,
    )
    return out