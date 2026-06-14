import torch
import triton
import triton.language as tl


@triton.jit
def _embedding_kernel(
    weight_ptr,
    idx_ptr,
    out_ptr,
    N,
    D,
    stride_w0,
    stride_w1,
    stride_o0,
    stride_o1,
    BLOCK_D: tl.constexpr,
):
    # Each program handles exactly one output row
    pid = tl.program_id(0)
    if pid >= N:
        return

    # Load the vocabulary index for this row
    idx_val = tl.load(idx_ptr + pid)

    # Base pointer to the weight row
    weight_row_base = weight_ptr + idx_val * stride_w0

    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    # Gather the embedding vector (random access, stream not reused)
    w = tl.load(weight_row_base + offs * stride_w1,
                mask=mask, other=0.0,
                eviction_policy="evict_first")

    # Store to output row
    out_row_base = out_ptr + pid * stride_o0
    tl.store(out_row_base + offs * stride_o1, w, mask=mask)


def triton_run(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Signature: triton_run(weight: f32[131072, 1024], idx: i64[1048576])
                -> f32[1048576, 1024]
    """
    assert weight.dtype == torch.float32
    assert idx.dtype == torch.int64
    assert weight.is_contiguous()

    N = idx.numel()          # 1048576
    D = weight.shape[1]      # 1024 (fixed per operator signature)

    out = torch.empty(N, D, dtype=weight.dtype, device=weight.device)

    grid = (N,)
    _embedding_kernel[grid](
        weight, idx, out,
        N, D,
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_D=D,
        num_warps=4,
        num_stages=2,
    )
    return out