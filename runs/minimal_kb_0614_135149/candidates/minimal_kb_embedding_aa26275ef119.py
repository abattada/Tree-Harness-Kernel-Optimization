import torch
import triton
import triton.language as tl


@triton.jit
def embedding_kernel(
    weight_ptr,
    idx_ptr,
    out_ptr,
    N,                   # total number of output rows
    D,                   # embedding dimension (only used for mask)
    stride_weight_0,     # leading dim of weight (D for contiguous weight)
    stride_weight_1,     # inner dim of weight (1 for contiguous weight)
    stride_out_0,        # leading dim of output (D for contiguous output)
    stride_out_1,        # inner dim of output (1 for contiguous output)
    ROWS_PER_PROG: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    start_row = pid * ROWS_PER_PROG

    # Offsets along the embedding dimension (compile-time constant BLOCK_D)
    offs_d = tl.arange(0, BLOCK_D)
    # Mask only needed if D < BLOCK_D; here D == BLOCK_D == 1024, but keep mask
    mask = offs_d < D

    # Process a batch of ROWS_PER_PROG consecutive rows
    for i in tl.static_range(ROWS_PER_PROG):
        row = start_row + i

        # Load the index for this row (once per row)
        idx = tl.load(idx_ptr + row, eviction_policy='evict_first')

        # Base pointer to the selected weight row and the output row
        weight_row_base = weight_ptr + idx * stride_weight_0
        out_row_base = out_ptr + row * stride_out_0

        # Load the full embedding row (contiguous, evict_first because no reuse)
        w = tl.load(weight_row_base + offs_d * stride_weight_1,
                    mask=mask, other=0.0,
                    eviction_policy='evict_first')

        # Store to the output row with streaming (non-temporal) write
        tl.store(out_row_base + offs_d * stride_out_1,
                 w, mask=mask, cache_modifier='cg')


def triton_run(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Gather embedding rows: out[i] = weight[idx[i]]
    weight: [131072, 1024] float32
    idx:    [1048576] int64
    returns: [1048576, 1024] float32
    """
    assert weight.dtype == torch.float32, "weight must be float32"
    assert idx.dtype == torch.int64, "idx must be int64"
    assert weight.is_contiguous(), "weight must be contiguous"
    assert idx.is_contiguous(), "idx must be contiguous"

    V, D = weight.shape
    N = idx.shape[0]
    assert D == 1024, f"Expected D=1024, got {D}"

    # Process multiple rows per program to amortize launch overhead.
    ROWS_PER_PROG = 8
    assert N % ROWS_PER_PROG == 0, (
        f"N={N} not divisible by ROWS_PER_PROG={ROWS_PER_PROG}"
    )
    grid = (N // ROWS_PER_PROG,)

    out = torch.empty(N, D, dtype=torch.float32, device=weight.device)

    embedding_kernel[grid](
        weight, idx, out,
        N, D,
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        ROWS_PER_PROG=ROWS_PER_PROG,
        BLOCK_D=D,           # full embedding dimension as compile-time constant
        num_warps=4,
        num_stages=2,
    )

    return out