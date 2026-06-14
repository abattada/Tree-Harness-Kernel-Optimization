import torch
import triton
import triton.language as tl

@triton.jit
def _embedding_kernel(
    weight_ptr,
    idx_ptr,
    out_ptr,
    N,
    D,
    stride_w0,
    stride_w1,
    stride_o0,
    stride_o1,
    ROWS_PER_PROG: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * ROWS_PER_PROG
    off_d = tl.arange(0, BLOCK_D)
    mask_d = off_d < D

    for i in range(ROWS_PER_PROG):
        row = start + i
        # Skip rows beyond N – no break needed
        if row < N:
            idx_val = tl.load(idx_ptr + row)
            w_row = tl.load(
                weight_ptr + idx_val * stride_w0 + off_d * stride_w1,
                mask=mask_d,
                other=0.0,
            )
            tl.store(
                out_ptr + row * stride_o0 + off_d * stride_o1,
                w_row,
                mask=mask_d,
            )


def triton_run(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Signature: triton_run(weight: f32[131072, 1024], idx: i64[1048576])
                -> f32[1048576, 1024]
    """
    assert weight.dtype == torch.float32
    assert idx.dtype == torch.int64
    assert weight.is_contiguous()
    N = idx.numel()
    D = weight.shape[1]

    out = torch.empty(N, D, dtype=weight.dtype, device=weight.device)

    ROWS_PER_PROG = 128
    BLOCK_D = D  # 1024 – known at compile time

    grid = ((N + ROWS_PER_PROG - 1) // ROWS_PER_PROG,)

    _embedding_kernel[grid](
        weight, idx, out,
        N, D,
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        ROWS_PER_PROG=ROWS_PER_PROG,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=2,
    )
    return out