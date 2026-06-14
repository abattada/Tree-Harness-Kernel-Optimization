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
    if pid >= N:
        return

    # load index
    idx = tl.load(idx_ptr + pid)

    # base pointer to weight row
    weight_row_base = weight_ptr + idx * stride_weight_0

    # offsets along embedding dimension
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < D

    # load weight row
    w = tl.load(weight_row_base + offsets * stride_weight_1, mask=mask, other=0.0)

    # base pointer to output row
    out_row_base = out_ptr + pid * stride_out_0
    tl.store(out_row_base + offsets * stride_out_1, w, mask=mask)


def triton_run(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    assert weight.dtype == torch.float32
    assert idx.dtype == torch.int64
    assert weight.is_contiguous()
    vocab, D = weight.shape
    N = idx.shape[0]
    assert D == 1024  # given in signature, but keep flexible

    out = torch.empty(N, D, dtype=torch.float32, device=weight.device)

    grid = (N,)
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