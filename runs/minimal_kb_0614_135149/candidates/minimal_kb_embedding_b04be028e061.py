import torch
import triton
import triton.language as tl


@triton.jit
def embedding_kernel(
    weight_ptr,       # fp32, shape [V, D]
    idx_ptr,          # i64, shape [N]
    out_ptr,          # fp32, shape [N, D]
    N,                # total number of output rows
    D,                # embedding dimension (=1024)
    stride_weight_0,  # leading dim of weight (D if contiguous)
    stride_weight_1,  # inner dim of weight (1 if contiguous)
    stride_out_0,     # leading dim of out (D if contiguous)
    stride_out_1,     # inner dim of out (1 if contiguous)
    ROWS_PER_PROG: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Process ROWS_PER_PROG consecutive output rows per program.
    Reduces grid size and amortizes launch overhead while increasing
    instruction-level parallelism.
    """
    pid = tl.program_id(0)
    start_row = pid * ROWS_PER_PROG

    # Offsets along embedding dimension
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < D

    for i in range(ROWS_PER_PROG):
        row = start_row + i

        # Load the vocabulary index for this output row
        idx = tl.load(idx_ptr + row)

        # Base pointers for the selected weight row and output row
        weight_row_base = weight_ptr + idx * stride_weight_0
        out_row_base = out_ptr + row * stride_out_0

        # Load the full embedding row with streaming cache hint
        w = tl.load(
            weight_row_base + offsets * stride_weight_1,
            mask=mask,
            other=0.0,
            eviction_policy='evict_first',
        )

        # Write the output row
        tl.store(
            out_row_base + offsets * stride_out_1,
            w,
            mask=mask,
        )


def triton_run(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather rows: out[i] = weight[idx[i]]"""
    assert weight.dtype == torch.float32, "weight must be float32"
    assert idx.dtype == torch.int64, "idx must be int64"
    assert weight.is_contiguous(), "weight must be contiguous"
    assert idx.is_contiguous(), "idx must be contiguous"
    V, D = weight.shape
    N = idx.shape[0]
    assert D == 1024, f"expected D=1024, got {D}"

    ROWS_PER_PROG = 8
    assert N % ROWS_PER_PROG == 0, "N must be divisible by ROWS_PER_PROG"

    out = torch.empty(N, D, dtype=torch.float32, device=weight.device)

    grid = (N // ROWS_PER_PROG,)

    embedding_kernel[grid](
        weight, idx, out,
        N, D,
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        ROWS_PER_PROG=ROWS_PER_PROG,
        BLOCK_D=D,             # compile-time specialization of full dim
        num_warps=4,
        num_stages=2,
    )

    return out