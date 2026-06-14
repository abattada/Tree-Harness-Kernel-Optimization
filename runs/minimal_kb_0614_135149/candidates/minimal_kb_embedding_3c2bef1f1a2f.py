import torch
import triton
import triton.language as tl


@triton.jit
def embedding_kernel(
    weight_ptr,
    idx_ptr,
    out_ptr,
    N,
    D,
    stride_weight_0,
    stride_weight_1,
    stride_out_0,
    stride_out_1,
    BLOCK_D: tl.constexpr,
):
    """Gather rows: out[i] = weight[idx[i]] – one program per row."""
    pid = tl.program_id(0)
    if pid >= N:
        return

    # Load the integer index for this output row
    idx = tl.load(idx_ptr + pid)

    # Base pointer to the corresponding embedding row
    weight_row_base = weight_ptr + idx * stride_weight_0

    # Offsets along the embedding dimension
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < D

    # Load the whole embedding row
    w = tl.load(weight_row_base + offsets * stride_weight_1, mask=mask, other=0.0)

    # Store into output
    out_row_base = out_ptr + pid * stride_out_0
    tl.store(out_row_base + offsets * stride_out_1, w, mask=mask)


def triton_run(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Gather embedding rows.
    Signature: triton_run(weight: f32[131072, 1024], idx: i64[1048576]) -> f32[1048576, 1024]
    """
    assert weight.dtype == torch.float32
    assert idx.dtype == torch.int64
    assert weight.is_contiguous(), "weight must be contiguous for coalesced access"
    vocab, D = weight.shape
    N = idx.numel()
    assert D == 1024, "Embedding dimension must be 1024"

    out = torch.empty(N, D, dtype=torch.float32, device=weight.device)

    grid = (N,)  # one program per output row
    embedding_kernel[grid](
        weight, idx, out,
        N, D,
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_D=1024,
        num_warps=4,
        num_stages=2,
    )

    return out