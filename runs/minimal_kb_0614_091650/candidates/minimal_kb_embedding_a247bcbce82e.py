import torch
import triton
import triton.language as tl

@triton.jit
def embedding_kernel(
    weight_ptr, idx_ptr, out_ptr,
    N, D,
    stride_weight_0, stride_weight_1,
    stride_out_0, stride_out_1,
    BATCH: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Each program processes BATCH consecutive rows from idx
    pid = tl.program_id(0)
    start = pid * BATCH
    end = tl.minimum(start + BATCH, N)

    # Offsets for the full embedding dimension (no mask needed when D == BLOCK_D)
    offs_d = tl.arange(0, BLOCK_D)

    for i in range(start, end):
        # Load the index
        idx = tl.load(idx_ptr + i)

        # Base pointer for the selected weight row
        weight_row_base = weight_ptr + idx * stride_weight_0

        # Load the full weight row (1024 contiguous floats)
        w = tl.load(weight_row_base + offs_d * stride_weight_1)

        # Base pointer for the output row
        out_row_base = out_ptr + i * stride_out_0
        tl.store(out_row_base + offs_d * stride_out_1, w)


def triton_run(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    assert weight.dtype == torch.float32
    assert idx.dtype == torch.int64
    assert weight.is_contiguous()
    vocab, D = weight.shape
    N = idx.shape[0]
    # D is fixed to 1024 per the signature
    assert D == 1024

    out = torch.empty(N, D, dtype=torch.float32, device=weight.device)

    # Process 64 rows per program to reduce grid size from 1M to ~16K
    BATCH: int = 64
    grid = ( (N + BATCH - 1) // BATCH, )
    embedding_kernel[grid](
        weight, idx, out,
        N, D,
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BATCH,
        BLOCK_D=D,
        num_warps=4,
        num_stages=2,
    )

    return out