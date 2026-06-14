import torch
import triton
import triton.language as tl

@triton.jit
def embedding_kernel(
    weight_ptr,          # f32[vocab, D]
    idx_ptr,             # i64[N]
    out_ptr,             # f32[N, D]
    N: tl.constexpr,
    D: tl.constexpr,     # embedding dimension (1024)
    stride_w0,           # = D (first stride of weight)
    stride_w1,           # = 1 (second stride of weight)
    stride_o0,           # = D (first stride of output)
    stride_o1,           # = 1 (second stride of output)
    BLOCK_D: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * ROWS_PER_PROG
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    for i in range(ROWS_PER_PROG):
        row = start + i
        if row >= N:
            break
        # load index
        idx = tl.load(idx_ptr + row)
        # load weight row: weight[idx, :]
        w = tl.load(weight_ptr + idx * stride_w0 + offs_d * stride_w1,
                    mask=mask_d, other=0.0, eviction_policy='evict_first')
        # store output row (contiguous in the D dimension)
        tl.store(out_ptr + row * stride_o0 + offs_d * stride_o1,
                 w, mask=mask_d)


def triton_run(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    assert weight.dtype == torch.float32
    assert idx.dtype == torch.int64
    assert weight.is_contiguous()
    assert idx.is_contiguous()
    vocab, D = weight.shape
    N = idx.numel()
    out = torch.empty(N, D, dtype=torch.float32, device=weight.device)

    # Tuning knobs — sensible starting values
    ROWS_PER_PROG = 8
    NUM_WARPS = 4
    NUM_STAGES = 2
    BLOCK_D = triton.next_power_of_2(D)  # 1024

    grid = (triton.cdiv(N, ROWS_PER_PROG),)

    embedding_kernel[grid](
        weight, idx, out,
        N, D,
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_D,
        ROWS_PER_PROG,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return out