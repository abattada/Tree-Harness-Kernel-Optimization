import torch
import triton
import triton.language as tl


@triton.jit
def embedding_kernel(
    weight_ptr,       # fp32, shape [V, D]
    idx_ptr,          # i64, shape [N]
    out_ptr,          # fp32, shape [N, D]
    N,                # number of output rows
    D,                # embedding dimension
    stride_weight_0,  # leading dim of weight (D if contiguous)
    stride_weight_1,  # inner dim of weight (1 if contiguous)
    stride_out_0,     # leading dim of out (D if contiguous)
    stride_out_1,     # inner dim of out (1 if contiguous)
    BLOCK_D: tl.constexpr,   # compile-time tile size along D
):
    pid = tl.program_id(0)
    if pid >= N:
        return

    # load the index for this output row
    idx = tl.load(idx_ptr + pid)

    # base pointers for the selected weight row and output row
    weight_row_base = weight_ptr + idx * stride_weight_0
    out_row_base = out_ptr + pid * stride_out_0

    # compute offsets along the embedding dimension
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < D

    # load the full embedding row (automatically split into multiple loads)
    w = tl.load(weight_row_base + offsets * stride_weight_1,
                mask=mask, other=0.0)

    # store to the output row
    tl.store(out_row_base + offsets * stride_out_1,
             w, mask=mask)


def triton_run(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather rows: out[i] = weight[idx[i]]"""
    assert weight.dtype == torch.float32, "weight must be float32"
    assert idx.dtype == torch.int64, "idx must be int64"
    assert weight.is_contiguous(), "weight must be contiguous"
    assert idx.is_contiguous(), "idx must be contiguous"
    V, D = weight.shape
    N = idx.shape[0]
    assert D == 1024, f"Expected D=1024, got {D}"

    # settings
    BLOCK_D = 1024      # full embedding dimension
    num_warps = 4
    num_stages = 2

    grid = (N,)
    out = torch.empty(N, D, dtype=torch.float32, device=weight.device)

    embedding_kernel[grid](
        weight, idx, out,
        N, D,
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return out