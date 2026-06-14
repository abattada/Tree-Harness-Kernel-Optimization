import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel: compute per‑row mean and population (unbiased=False) variance
# for a 2‑D float32 tensor.
# ---------------------------------------------------------------------------
#
# Launch parameters (tunable knobs):
#   BLOCK_SIZE = 4096   (must equal n_cols; no masking)
#   num_warps  = 8
#   num_stages = 4
#
# The input shape (8192, 4096) lets us assign exactly one block per row,
# making loads and stores fully vectorized and contention‑free.
# ---------------------------------------------------------------------------

@triton.jit
def welford_kernel(
    x_ptr,                             # [n_rows, n_cols] contiguous input
    out_ptr,                           # [2, n_rows] output (row0=mean, row1=var)
    n_cols: int,                       # 4096 (static, but not constexpr)
    n_rows: int,                       # 8192 (used for output strides)
    BLOCK_SIZE: tl.constexpr,          # must equal n_cols (4096)
):
    pid = tl.program_id(0)              # row index

    # Pointer to the start of the row
    row_start = pid * n_cols

    # Load the whole row (unaligned fine – all floats are contiguous)
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + row_start + offsets)

    # Sum and sum of squares in float32 (single‑pass, no welford needed)
    s = tl.sum(x, axis=0).to(tl.float32)
    sq = tl.sum(x * x, axis=0).to(tl.float32)

    mean = s / n_cols
    var  = (sq / n_cols) - mean * mean    # population variance

    # Store results – output layout: [2, n_rows]
    tl.store(out_ptr + 0 * n_rows + pid, mean)
    tl.store(out_ptr + 1 * n_rows + pid, var)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    Equivalent to:
        mean = x.mean(dim=-1)
        var  = x.var(dim=-1, unbiased=False)
        return torch.stack([mean, var])

    Input:  x shape (8192, 4096), dtype float32, CUDA device
    Output: out shape (2, 8192), dtype float32
    """
    assert x.is_cuda and x.dtype == torch.float32
    n_rows, n_cols = x.shape
    assert n_cols == 4096, "This kernel requires n_cols = 4096"

    out = torch.empty((2, n_rows), dtype=torch.float32, device=x.device)

    # Block size must equal the full row length to avoid masking.
    BLOCK_SIZE = n_cols  # 4096

    grid = (n_rows,)
    welford_kernel[grid](
        x, out,
        n_cols, n_rows,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,      # tunable
        num_stages=4,     # tunable
    )
    return out