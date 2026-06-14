import torch
import triton
import triton.language as tl


@triton.jit
def embedding_kernel(
    weight_ptr,
    idx_ptr,
    out_ptr,
    N,
    stride_weight_0,
    stride_weight_1,
    stride_out_0,
    stride_out_1,
    D: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    """Each program handles ROWS_PER_PROG output rows; one wide load per row."""
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROG

    # Offsets along the embedding dimension (D is constexpr, so arange is fully known)
    offs_d = tl.arange(0, D)

    for i in range(ROWS_PER_PROG):
        row = row_start + i
        if row < N:
            # Load the vocabulary index
            idx = tl.load(idx_ptr + row)

            # Base pointers for the selected weight row and the output row
            weight_base = weight_ptr + idx * stride_weight_0
            out_base = out_ptr + row * stride_out_0

            # Load the whole row in one coalesced transaction (no mask needed:
            # D == weight.shape[1] and the row is contiguous)
            w = tl.load(weight_base + offs_d * stride_weight_1)

            # Store the row
            tl.store(out_base + offs_d * stride_out_1, w)


def triton_run(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather rows: out[i] = weight[idx[i]]

    Args:
        weight: float32 tensor of shape [131072, 1024] (vocab, dim)
        idx:    int64 tensor of shape [1048576] containing valid indices

    Returns:
        out: float32 tensor of shape [1048576, 1024]
    """
    assert weight.dtype == torch.float32, "weight must be float32"
    assert idx.dtype == torch.int64, "idx must be int64"
    assert weight.is_contiguous(), "weight must be contiguous"
    assert idx.is_contiguous(), "idx must be contiguous"
    vocab, D = weight.shape
    N = idx.shape[0]
    assert D == 1024, f"Expected D=1024, got {D}"

    # Number of rows each program processes. Must divide N exactly.
    ROWS_PER_PROG = 8
    assert N % ROWS_PER_PROG == 0, (
        f"N ({N}) must be divisible by ROWS_PER_PROG ({ROWS_PER_PROG})"
    )

    grid = (N // ROWS_PER_PROG,)
    out = torch.empty(N, D, dtype=torch.float32, device=weight.device)

    embedding_kernel[grid](
        weight, idx, out,
        N,
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        D=D,                      # constexpr, eliminates the inner tile loop
        ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=4,
        num_stages=2,
    )

    return out