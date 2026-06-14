import torch
import triton
import triton.language as tl


@triton.jit
def _embedding_kernel(
    weight_ptr,
    idx_ptr,
    out_ptr,
    N,
    stride_w0,
    stride_w1,
    stride_o0,
    stride_o1,
    ROWS_PER_PROG: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROG
    off_d = tl.arange(0, BLOCK_D)

    # ROWS_PER_PROG divides N exactly, so no inner boundary check needed.
    for i in range(ROWS_PER_PROG):
        row = row_start + i
        idx_val = tl.load(idx_ptr + row)

        # Gather a full row from the weight matrix.
        w_row = tl.load(
            weight_ptr + idx_val * stride_w0 + off_d * stride_w1,
            eviction_policy="evict_first",
        )
        tl.store(
            out_ptr + row * stride_o0 + off_d * stride_o1,
            w_row,
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
    assert D == 1024, "Embedding dimension must be 1024 as per signature"

    # Choose a power-of-two rows-per-program that divides N exactly.  This
    # amortises launch overhead without introducing partial rows inside the
    # kernel, so we can drop the per-row boundary test.
    ROWS_PER_PROG = 256
    assert N % ROWS_PER_PROG == 0, (
        f"N ({N}) must be divisible by ROWS_PER_PROG ({ROWS_PER_PROG})"
    )

    out = torch.empty(N, D, dtype=weight.dtype, device=weight.device)

    grid = (N // ROWS_PER_PROG,)
    BLOCK_D = D  # 1024

    _embedding_kernel[grid](
        weight, idx, out,
        N,
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        ROWS_PER_PROG=ROWS_PER_PROG,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=2,
    )
    return out