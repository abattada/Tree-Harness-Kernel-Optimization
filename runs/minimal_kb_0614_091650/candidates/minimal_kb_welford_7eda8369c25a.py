import torch
import triton
import triton.language as tl

# Shape is fixed for this operator: 8192 rows x 4096 cols.
# We exploit this fully to provide compile-time constants.
N_ROWS = 8192
N_COLS = 4096
BLOCK_SIZE = N_COLS          # entire row fits in one block, no masking
NUM_WARPS = 4
NUM_STAGES = 4
INV_N = 1.0 / N_COLS         # compile-time reciprocal for faster division

@triton.jit
def welford_kernel(
    x_ptr,                  # [N_ROWS, N_COLS] input
    out_ptr,                # [2, N_ROWS] output (row0=mean, row1=var)
    n_rows: tl.constexpr,   # 8192
    n_cols: tl.constexpr,   # 4096
    BLOCK_SIZE: tl.constexpr,
    INV_N: tl.constexpr,    # 1/4096
):
    pid = tl.program_id(0)          # row index
    row_start = pid * n_cols

    # Load the entire row with hints for vectorization and eviction policy.
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
    x_ptrs = x_ptr + row_start + offsets
    # Use evict_first since the input is read only once.
    x = tl.load(x_ptrs, eviction_policy='evict_first')

    # Single-pass sum and sum of squares.
    s  = tl.sum(x, axis=0).to(tl.float32)
    sq = tl.sum(x * x, axis=0).to(tl.float32)

    # Compute mean and population variance using compile-time inverse.
    mean = s * INV_N
    # var = (sq / n_cols) - mean * mean
    var = (sq * INV_N) - mean * mean

    # Store to output [2, n_rows]
    tl.store(out_ptr + 0 * n_rows + pid, mean)
    tl.store(out_ptr + 1 * n_rows + pid, var)

def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32
    assert x.shape == (N_ROWS, N_COLS), f"Expected ({N_ROWS}, {N_COLS}) but got {x.shape}"

    out = torch.empty(2, N_ROWS, dtype=torch.float32, device=x.device)

    grid = (N_ROWS,)
    welford_kernel[grid](
        x, out,
        n_rows=N_ROWS,
        n_cols=N_COLS,
        BLOCK_SIZE=BLOCK_SIZE,
        INV_N=INV_N,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return out