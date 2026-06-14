import torch
import triton
import triton.language as tl


@triton.jit
def embedding_kernel(
    weight_ptr,       # fp32, shape [V, D]
    idx_ptr,          # i64, shape [N]
    out_ptr,          # fp32, shape [N, D]
    N,                # number of output rows
    D,                # embedding dimension
    stride_weight_0,  # leading dim of weight (D if contiguous)
    stride_weight_1,  # inner dim of weight (1 if contiguous)
    stride_out_0,     # leading dim of out (D if contiguous)
    stride_out_1,     # inner dim of out (1 if contiguous)
    BLOCK_D: tl.constexpr,       # tile size along embedding dimension
    ROWS_PER_PROG: tl.constexpr, # how many output rows each program handles
):
    # Each program processes ROWS_PER_PROG consecutive output rows.
    # For each row it loads the index, then tiles along the embedding dim
    # in blocks of BLOCK_D (with boundary masking for the last block).
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROG

    num_d_blocks = tl.cdiv(D, BLOCK_D)

    for i in tl.static_range(ROWS_PER_PROG):
        row = row_start + i
        if row < N:
            # Load the index for this output row
            idx = tl.load(idx_ptr + row)

            # Base pointers for the weight row and output row
            weight_row_base = weight_ptr + idx * stride_weight_0
            out_row_base = out_ptr + row * stride_out_0

            # Tile along the embedding dimension
            for d_start in tl.static_range(0, num_d_blocks):
                offs_d = d_start * BLOCK_D + tl.arange(0, BLOCK_D)
                mask = offs_d < D
                w = tl.load(weight_row_base + offs_d * stride_weight_1,
                            mask=mask, other=0.0)
                tl.store(out_row_base + offs_d * stride_out_1,
                         w, mask=mask)


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
    V, D = weight.shape
    N = idx.shape[0]
    assert D == 1024, f"Expected D=1024, got {D}"

    # Tuning knobs (exposed for later optimization)
    BLOCK_D = 256          # try: 128, 256, 512, 1024
    ROWS_PER_PROG = 4      # try: 1, 2, 4, 8, 16
    NUM_WARPS = 4          # try: 2, 4, 8
    NUM_STAGES = 2         # try: 2, 3, 4

    # Ensure N is divisible by ROWS_PER_PROG to avoid a slow boundary loop.
    # In practice we could always add a final partial block, but here we
    # assert exact division for simplicity.
    assert N % ROWS_PER_PROG == 0, (
        f"N ({N}) must be divisible by ROWS_PER_PROG ({ROWS_PER_PROG}) "
        f"for this simple launch. Use a boundary loop for general N."
    )

    grid = (N // ROWS_PER_PROG,)

    out = torch.empty(N, D, dtype=torch.float32, device=weight.device)

    embedding_kernel[grid](
        weight, idx, out,
        N, D,
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_D=BLOCK_D,
        ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )

    return out