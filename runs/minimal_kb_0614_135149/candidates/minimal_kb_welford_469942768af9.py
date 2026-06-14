import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Persistent kernel : each program processes multiple rows in a grid‑stride
# loop, amortising launch overhead.  The reduction per row is a single‑pass
# sum/sum‑of‑squares, producing population mean and variance.
# ---------------------------------------------------------------------------

N_ROWS = 8192
N_COLS = 4096
BLOCK_SIZE = N_COLS
INV_N = 1.0 / N_COLS          # compile‑time reciprocal to replace division

@triton.jit
def welford_persistent_kernel(
    x_ptr,                     # input [N_ROWS, N_COLS] float32
    out_ptr,                   # output [2, N_ROWS] float32
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INV_N: tl.constexpr,
    GRID_SIZE: tl.constexpr,   # total number of launched programs
):
    pid = tl.program_id(0)

    # Grid‑stride loop: each program handles rows pid, pid+GRID_SIZE, ...
    for row_idx in range(pid, n_rows, GRID_SIZE):
        row_start = row_idx * n_cols
        offsets = tl.arange(0, BLOCK_SIZE)
        x_ptrs = x_ptr + row_start + offsets

        # Load the whole row in one contiguous, coalesced transaction
        x = tl.load(x_ptrs, eviction_policy='evict_first')

        # Single‑pass sum and sum of squares in fp32
        s  = tl.sum(x, axis=0).to(tl.float32)
        sq = tl.sum(x * x, axis=0).to(tl.float32)

        mean = s * INV_N
        var = (sq * INV_N) - mean * mean   # population variance

        # Store the two statistics into the output [2, n_rows] layout
        out_stride = n_rows
        tl.store(out_ptr + 0 * out_stride + row_idx, mean)
        tl.store(out_ptr + 1 * out_stride + row_idx, var)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    Compute per‑row mean and population variance of a float32 tensor.

    Equivalent to:
        mean = x.mean(dim=-1)
        var  = x.var(dim=-1, unbiased=False)
        return torch.stack([mean, var])

    Input:  x shape (8192, 4096), float32, CUDA device.
    Output: out shape (2, 8192), float32.
    """
    assert x.is_cuda and x.dtype == torch.float32
    assert x.shape == (N_ROWS, N_COLS), f"Expected ({N_ROWS}, {N_COLS}) but got {x.shape}"

    out = torch.empty((2, N_ROWS), dtype=torch.float32, device=x.device)

    # Grid size reduced to avoid launching a block per row.
    # 1024 persistent blocks keeps all SMs busy while cutting launch overhead.
    GRID_SIZE = 1024
    grid = (GRID_SIZE,)

    welford_persistent_kernel[grid](
        x, out,
        n_rows=N_ROWS,
        n_cols=N_COLS,
        BLOCK_SIZE=BLOCK_SIZE,
        INV_N=INV_N,
        GRID_SIZE=GRID_SIZE,
        num_warps=4,       # fewer warps per block → higher occupancy with persistent loops
        num_stages=2,      # no pipelining needed; low stages keep register pressure down
    )
    return out