import torch
import triton
import triton.language as tl


@triton.jit
def embed_kernel(
    weight_ptr, idx_ptr, out_ptr,
    N, D,
    weight_stride, idx_stride, out_stride,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    """
    Kernel for embedding lookup: out[i] = weight[idx[i]].
    Each program handles `ROWS_PER_PROG` output rows.
    Loads a full row of `weight` (D elements) for each index.
    """
    pid = tl.program_id(0)
    start_row = pid * ROWS_PER_PROG
    offs_d = tl.arange(0, BLOCK_SIZE)

    # Loop over rows assigned to this program
    for r in range(ROWS_PER_PROG):
        row = start_row + r
        if row < N:
            # Load index (i64)
            idx_val = tl.load(idx_ptr + row * idx_stride)  # idx_stride is 1
            # Compute base pointer into weight for this row
            w_base = weight_ptr + idx_val * D
            # Load entire weight row (no mask because BLOCK_SIZE == D)
            w_row = tl.load(w_base + offs_d, eviction_policy='evict_first')
            # Store to output row
            o_base = out_ptr + row * D
            tl.store(o_base + offs_d, w_row, eviction_policy='evict_first')


def triton_run(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    weight: f32[131072, 1024]
    idx: i64[1048576]
    Returns: f32[1048576, 1024]
    """
    assert weight.dtype == torch.float32
    assert idx.dtype == torch.int64
    N = idx.shape[0]          # number of output rows
    D = weight.shape[1]       # embedding dimension

    output = torch.empty((N, D), dtype=torch.float32, device=weight.device)

    # Launch parameters
    BLOCK_SIZE = D  # load full row in one contiguous block
    ROWS_PER_PROG = 64  # tunable, balances grid size and register pressure
    grid = ((N + ROWS_PER_PROG - 1) // ROWS_PER_PROG,)

    embed_kernel[grid](
        weight.data_ptr(), idx.data_ptr(), output.data_ptr(),
        N, D,
        weight.stride(0), idx.stride(0), output.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        ROWS_PER_PROG=ROWS_PER_PROG,
    )

    return output