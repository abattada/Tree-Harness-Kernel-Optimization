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
    ROWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    start_row = pid * ROWS_PER_PROG
    for row_off in range(ROWS_PER_PROG):
        current_row = start_row + row_off
        if current_row >= N:
            break

        # load index for this row
        idx = tl.load(idx_ptr + current_row, eviction_policy='evict_first')

        # base pointer to the selected weight row
        weight_row_base = weight_ptr + idx * stride_weight_0

        # offsets for dimension D
        offsets = tl.arange(0, BLOCK_D)
        mask = offsets < D

        # load the full weight row (coalesced)
        w = tl.load(
            weight_row_base + offsets * stride_weight_1,
            mask=mask,
            other=0.0,
            eviction_policy='evict_first',
        )

        # store to output row
        out_row_base = out_ptr + current_row * stride_out_0
        tl.store(out_row_base + offsets * stride_out_1, w, mask=mask)


def triton_run(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    assert weight.dtype == torch.float32
    assert idx.dtype == torch.int64
    # ensure weight is contiguous (already from signature)
    assert weight.is_contiguous()
    vocab, D = weight.shape
    N = idx.shape[0]
    assert D == 1024  # given in the problem signature

    out = torch.empty(N, D, dtype=torch.float32, device=weight.device)

    # process 4 rows per program to reduce grid size and launch overhead
    ROWS_PER_PROG = 4
    grid = (triton.cdiv(N, ROWS_PER_PROG),)

    embedding_kernel[grid](
        weight, idx, out,
        N, D,
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_D=D,
        ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=4,
        num_stages=2,
    )

    return out