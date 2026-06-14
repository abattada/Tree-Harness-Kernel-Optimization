import torch
import triton
import triton.language as tl

@triton.jit
def embedding_kernel(
    weight_ptr, idx_ptr, out_ptr,
    N, D,
    stride_weight_0, stride_weight_1,
    stride_out_0, stride_out_1,
    ROWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    start_row = pid * ROWS_PER_PROG
    offs = tl.arange(0, D)
    mask = offs < D

    for i in range(ROWS_PER_PROG):
        row = start_row + i
        if row >= N:
            break

        # load index
        idx = tl.load(idx_ptr + row)

        # base pointer to weight row
        weight_row_ptr = weight_ptr + idx * stride_weight_0

        # load weight row (streaming, evict_first)
        w = tl.load(weight_row_ptr + offs * stride_weight_1, mask=mask, other=0.0, eviction_policy='evict_first')

        # store output row (streaming, evict_first)
        out_row_ptr = out_ptr + row * stride_out_0
        tl.store(out_row_ptr + offs * stride_out_1, w, mask=mask, eviction_policy='evict_first')


def triton_run(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    assert weight.dtype == torch.float32
    assert idx.dtype == torch.int64
    assert weight.is_contiguous()
    vocab, D = weight.shape
    N = idx.shape[0]
    assert D == 1024  # given in signature, but keep flexible

    out = torch.empty(N, D, dtype=torch.float32, device=weight.device)

    # Process 8 rows per program to amortize launch overhead and improve occupancy
    ROWS_PER_PROG = 8
    grid = ( (N + ROWS_PER_PROG - 1) // ROWS_PER_PROG, )  # ceil division

    embedding_kernel[grid](
        weight, idx, out,
        N, D,
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        ROWS_PER_PROG,
        num_warps=4,
        num_stages=2,
    )

    return out