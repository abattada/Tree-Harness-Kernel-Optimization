import torch
import triton
import triton.language as tl

# Fixed dimensions for this operator — exploited as tl.constexpr to enable
# aggressive compiler optimisations (constant folding, vectorisation).
N_ROWS    = 8192
N_COLS    = 4096
BLOCK_SZ  = N_COLS                # single block covers an entire row — no masks
INV_N     = 1.0 / float(N_COLS)   # compile‑time reciprocal avoids expensive sdiv

@triton.jit
def welford_kernel(
    x_ptr,                      # [N_ROWS, N_COLS] float32 input, row‑major
    out_ptr,                    # [2, N_ROWS] float32 output (row0 = mean, row1 = var)
    n_rows: tl.constexpr,        # 8192
    n_cols: tl.constexpr,        # 4096
    BLOCK_SIZE: tl.constexpr,    # = N_COLS
    INV_N: tl.constexpr,         # = 1/N_COLS
):
    pid = tl.program_id(0)            # row index
    row_start = pid * n_cols

    # Offsets for the whole row, hinting contiguous vectorised access.
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
    x_ptrs = x_ptr + row_start + offsets

    # Input is read exactly once — evict_first prevents cache pollution.
    x = tl.load(x_ptrs, eviction_policy='evict_first')

    # Single‑pass sum and sum‑of‑squares (both accumulated in fp32).
    s  = tl.sum(x,       axis=0).to(tl.float32)
    sq = tl.sum(x * x,   axis=0).to(tl.float32)

    # Population mean and variance using the compile‑time reciprocal.
    mean = s * INV_N
    var  = (sq * INV_N) - mean * mean

    # Write results: output array is [2, N_ROWS].
    tl.store(out_ptr + 0 * n_rows + pid, mean)
    tl.store(out_ptr + 1 * n_rows + pid, var)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    Per‑row mean and population variance of a float32 tensor.

    Equivalent to:
        mean = x.mean(dim=-1)
        var  = x.var(dim=-1, unbiased=False)
        return torch.stack([mean, var])

    Args:
        x: torch.Tensor of shape (8192, 4096), dtype float32, on CUDA.
    Returns:
        Out tensor of shape (2, 8192), dtype float32.
    """
    assert x.is_cuda and x.dtype == torch.float32
    n_rows, n_cols = x.shape
    # The kernel is specialised for this exact shape; any deviation is a contract violation.
    assert n_rows == N_ROWS and n_cols == N_COLS, \
        f"Expected ({N_ROWS}, {N_COLS}) but got ({n_rows}, {n_cols})"

    out = torch.empty((2, n_rows), dtype=torch.float32, device=x.device)

    grid = (n_rows,)
    welford_kernel[grid](
        x,
        out,
        n_rows=N_ROWS,
        n_cols=N_COLS,
        BLOCK_SIZE=BLOCK_SZ,
        INV_N=INV_N,
        num_warps=4,          # good trade‑off for memory‑bound reduction
        num_stages=2,         # minimal pipelining needed → lower register pressure
    )
    return out