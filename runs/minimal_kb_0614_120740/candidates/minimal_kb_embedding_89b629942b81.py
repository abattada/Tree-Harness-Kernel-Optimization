import torch
import triton
import triton.language as tl

@triton.jit
def embedding_kernel(
    weight_ptr, idx_ptr, out_ptr,
    D, stride_weight_0, stride_weight_1,
    stride_out_0, stride_out_1,
    N,
    BLOCK_D: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    start_row = pid * ROWS_PER_PROG
    # load weight and output rows in a loop
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    for i in range(ROWS_PER_PROG):
        row = start_row + i
        if row >= N:
            break
        idx = tl.load(idx_ptr + row, eviction_policy='evict_last')
        # weight row pointer
        w_ptr = weight_ptr + idx * stride_weight_0
        # load weight row
        w = tl.load(w_ptr + offs * stride_weight_1, mask=mask, other=0.0, eviction_policy='evict_first')
        # output row pointer
        out_ptr_row = out_ptr + row * stride_out_0
        tl.store(out_ptr_row + offs * stride_out_1, w, mask=mask)


def triton_run(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    assert weight.dtype == torch.float32
    assert idx.dtype == torch.int64
    assert weight.is_contiguous()
    vocab, D = weight.shape
    N = idx.shape[0]
    assert D == 1024  # kept flexible but known constant

    out = torch.empty(N, D, dtype=torch.float32, device=weight.device)

    ROWS_PER_PROG = 8
    grid = ((N + ROWS_PER_PROG - 1) // ROWS_PER_PROG,)

    embedding_kernel[grid](
        weight, idx, out,
        D,
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        N,
        BLOCK_D=D,
        ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=4,
        num_stages=2,
    )

    return out