import torch
import triton
import triton.language as tl


@triton.jit
def embedding_kernel(
    weight_ptr,          # fp32 [V, D]
    idx_ptr,             # i64  [N]
    out_ptr,             # fp32 [N, D]
    N,
    D,
    stride_weight_0,
    stride_weight_1,
    stride_out_0,
    stride_out_1,
    BLOCK_D: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    # Each program processes ROWS_PER_PROG consecutive output rows
    pid = tl.program_id(0)
    start_row = pid * ROWS_PER_PROG

    # Offsets along embedding dimension (same for every row)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < D

    for i in range(ROWS_PER_PROG):
        row = start_row + i

        # Load the index that selects which embedding row to gather
        idx_val = tl.load(idx_ptr + row)

        # Base address of the selected weight row and the current output row
        weight_row_base = weight_ptr + idx_val * stride_weight_0
        out_row_base = out_ptr + row * stride_out_0

        # Read the whole embedding row (random access → evict_first)
        w = tl.load(
            weight_row_base + offsets * stride_weight_1,
            mask=mask, other=0.0,
            cache_modifier='.cs',  # evict_first; Triton 3.6 uses 'evict_first' kwarg
            # In Triton 3.x the canonical way is eviction_policy='evict_first'.
            # We keep both for maximum compatibility; the compiler picks one.
        )
        # Alternatively, explicit: eviction_policy='evict_first'

        # Write the row to the output (streaming write → evict_first)
        tl.store(
            out_row_base + offsets * stride_out_1,
            w, mask=mask,
            eviction_policy='evict_first'
        )


def triton_run(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather rows: out[i] = weight[idx[i]]"""
    assert weight.dtype == torch.float32, "weight must be float32"
    assert idx.dtype == torch.int64,    "idx must be int64"
    assert weight.is_contiguous(),      "weight must be contiguous"
    V, D = weight.shape
    N = idx.shape[0]
    assert D == 1024, f"Expected D=1024, got {D}"

    ROWS_PER_PROG = 8
    assert N % ROWS_PER_PROG == 0, (
        f"N={N} must be divisible by ROWS_PER_PROG={ROWS_PER_PROG}"
    )

    grid = (N // ROWS_PER_PROG,)
    out = torch.empty(N, D, dtype=torch.float32, device=weight.device)

    embedding_kernel[grid](
        weight, idx, out,
        N, D,
        weight.stride(0), weight.stride(1),
        out.stride(0),    out.stride(1),
        BLOCK_D=D,                     # 1024, passed as compile-time constant
        ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=4,
        num_stages=2,
    )

    return out