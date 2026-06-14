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
    offsets = tl.arange(0, BLOCK_D)

    for i in range(ROWS_PER_PROG):
        row = start_row + i
        # load index for this row
        idx = tl.load(idx_ptr + row, eviction_policy='evict_first')

        # base pointer to the selected row in weight
        weight_base = weight_ptr + idx * stride_weight_0

        # load the full embedding row (BLOCK_D elements)
        w = tl.load(weight_base + offsets * stride_weight_1, eviction_policy='evict_first')

        # store to output row
        out_base = out_ptr + row * stride_out_0
        tl.store(out_base + offsets * stride_out_1, w)


def triton_run(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    assert weight.dtype == torch.float32
    assert idx.dtype == torch.int64
    assert weight.is_contiguous()
    assert idx.is_contiguous()
    vocab, D = weight.shape
    N = idx.shape[0]
    assert D == 1024  # fixed by signature, but general code works

    out = torch.empty(N, D, dtype=torch.float32, device=weight.device)

    ROWS_PER_PROG = 8
    assert N % ROWS_PER_PROG == 0, "N must be divisible by ROWS_PER_PROG for simplicity"
    grid = (N // ROWS_PER_PROG,)

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