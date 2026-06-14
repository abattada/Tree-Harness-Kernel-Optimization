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
    """
    Gather embedding rows: out[row, :] = weight[idx[row], :]
    Each program processes ROWS_PER_PROG consecutive rows.
    """
    pid = tl.program_id(0)
    start_row = pid * ROWS_PER_PROG
    offs_d = tl.arange(0, BLOCK_D)

    for i in range(ROWS_PER_PROG):
        row = start_row + i
        if row >= N:
            return

        # Load the index for this row
        idx = tl.load(idx_ptr + row)

        # Load the full embedding row
        weight_row_base = weight_ptr + idx * stride_weight_0
        w = tl.load(weight_row_base + offs_d * stride_weight_1, other=0.0)

        # Store to output
        out_row_base = out_ptr + row * stride_out_0
        tl.store(out_row_base + offs_d * stride_out_1, w)


def triton_run(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Embedding forward: out[i, :] = weight[idx[i], :]
    weight: [vocab_size, D] f32
    idx: [N] i64
    returns: [N, D] f32
    """
    assert weight.dtype == torch.float32
    assert idx.dtype == torch.int64
    assert weight.is_contiguous()
    vocab, D = weight.shape
    N = idx.shape[0]
    assert D == 1024, "This kernel is tuned for D=1024"

    out = torch.empty(N, D, dtype=torch.float32, device=weight.device)

    # Tunable parameters: ROWS_PER_PROG (e.g., 8) and num_warps (e.g., 4)
    ROWS_PER_PROG = 8
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