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
    stride_w0,
    stride_w1,
    stride_o0,
    stride_o1,
    BLOCK_D: tl.constexpr,
):
    # One program per output row, providing massive parallelism for a memory-bound gather.
    pid = tl.program_id(0)
    if pid >= N:
        return

    # Load the index for this row; idx is an int64 tensor.
    idx_val = tl.load(idx_ptr + pid)

    # Base pointer to the weight row inside the embedding table.
    weight_row_ptr = weight_ptr + idx_val * stride_w0

    # Offsets along the embedding dimension.
    offs_d = tl.arange(0, BLOCK_D)
    # Since D == BLOCK_D (1024), the mask is always true, but kept for safety.
    mask = offs_d < D

    # Gather the whole row in a single coalesced load.
    w = tl.load(
        weight_row_ptr + offs_d * stride_w1,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",  # streaming read, rows are not reused
    )

    # Base pointer for the output row.
    out_row_ptr = out_ptr + pid * stride_o0
    tl.store(
        out_row_ptr + offs_d * stride_o1,
        w,
        mask=mask,
    )


def triton_run(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Signature: triton_run(weight: f32[131072, 1024], idx: i64[1048576]) -> f32[1048576, 1024]
    """
    # Input validation
    assert weight.dtype == torch.float32, "weight must be float32"
    assert idx.dtype == torch.int64, "indices must be int64"
    assert weight.is_contiguous(), "weight must be contiguous"
    vocab, D = weight.shape
    N = idx.numel()
    assert D == 1024, f"Embedding dimension must be 1024, got {D}"

    # Allocate contiguous output tensor.
    out = torch.empty(N, D, dtype=torch.float32, device=weight.device)

    # Launch one program per output row to maximise parallelism and memory latency hiding.
    grid = (N,)
    embedding_kernel[grid](
        weight,
        idx,
        out,
        N,
        D,
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_D=D,          # compile-time constant for full-row load/store
        num_warps=4,        # enough warps for a 1024-element block
        num_stages=2,      # standard pipelining
    )
    return out