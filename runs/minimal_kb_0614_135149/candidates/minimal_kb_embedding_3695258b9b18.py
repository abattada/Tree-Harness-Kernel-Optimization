import torch
import triton
import triton.language as tl


@triton.jit
def embedding_kernel(
    weight_ptr,          # fp32, shape [V, D], contiguous
    idx_ptr,             # i64, shape [N], contiguous
    out_ptr,             # fp32, shape [N, D], contiguous
    N,                   # total number of output rows
    D,                   # embedding dimension (1024)
    stride_weight_0,     # leading dim of weight (D)
    stride_weight_1,     # inner dim of weight (1)
    stride_out_0,        # leading dim of out (D)
    stride_out_1,        # inner dim of out (1)
    ROWS_PER_PROG: tl.constexpr,   # rows processed per program
    BLOCK_D: tl.constexpr,         # D, so that we can vectorize fully
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROG

    # Load the indices for this program's batch of rows (coalesced)
    offs_row = tl.arange(0, ROWS_PER_PROG)
    idx_vals = tl.load(idx_ptr + row_start + offs_row)

    # Offsets along the embedding dimension
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    # Unrolled loop over the batch of rows
    for r in tl.static_range(ROWS_PER_PROG):
        idx = idx_vals[r]

        # Base pointers to the weight row and the output row
        w_base = weight_ptr + idx * stride_weight_0
        out_base = out_ptr + (row_start + r) * stride_out_0

        # Load the full weight row with streaming hint (no reuse expected)
        w = tl.load(w_base + offs_d * stride_weight_1, mask=mask_d, other=0.0,
                    eviction_policy='evict_first')

        # Store it directly to the output
        tl.store(out_base + offs_d * stride_out_1, w, mask=mask_d)


def triton_run(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather rows: out[i] = weight[idx[i]]"""
    assert weight.dtype == torch.float32, "weight must be float32"
    assert idx.dtype == torch.int64, "idx must be int64"
    assert weight.is_contiguous(), "weight must be contiguous"
    assert idx.is_contiguous(), "idx must be contiguous"

    V, D = weight.shape
    N = idx.shape[0]
    assert D == 1024, f"Expected D=1024, got {D}"

    ROWS_PER_PROG = 16          # process 16 rows per program to cut grid size
    assert N % ROWS_PER_PROG == 0, f"N ({N}) must be divisible by ROWS_PER_PROG"

    grid = (N // ROWS_PER_PROG,)
    out = torch.empty(N, D, dtype=torch.float32, device=weight.device)

    embedding_kernel[grid](
        weight, idx, out,
        N, D,
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        ROWS_PER_PROG=ROWS_PER_PROG,
        BLOCK_D=D,
        num_warps=8,            # increased warps for more in-flight memory requests
        num_stages=2,
    )

    return out