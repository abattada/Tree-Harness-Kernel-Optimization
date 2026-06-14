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
    base_row = pid * ROWS_PER_PROG

    offsets = tl.arange(0, BLOCK_D)

    for i in range(ROWS_PER_PROG):
        row = base_row + i
        if row >= N:
            break

        # load index
        idx = tl.load(idx_ptr + row, eviction_policy='evict_first')

        # base pointer to weight row
        weight_row_base = weight_ptr + idx * stride_weight_0

        # load weight row (no mask needed since BLOCK_D == D and offsets always valid)
        w = tl.load(weight_row_base + offsets * stride_weight_1,
                    eviction_policy='evict_first')

        # store output
        out_row_base = out_ptr + row * stride_out_0
        tl.store(out_row_base + offsets * stride_out_1, w)


def triton_run(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    assert weight.dtype == torch.float32
    assert idx.dtype == torch.int64
    assert weight.is_contiguous()
    assert weight.shape[1] == 1024  # given in signature, ensures no remainders
    vocab, D = weight.shape
    N = idx.shape[0]

    out = torch.empty(N, D, dtype=torch.float32, device=weight.device)

    # Process multiple rows per program to reduce launch overhead
    ROWS_PER_PROG = 16
    assert N % ROWS_PER_PROG == 0, "N must be divisible by ROWS_PER_PROG for optimal performance"
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