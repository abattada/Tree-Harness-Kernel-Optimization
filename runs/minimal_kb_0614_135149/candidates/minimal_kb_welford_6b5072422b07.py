import torch
import triton
import triton.language as tl

N_ROWS = 8192
N_COLS = 4096
BLOCK_SIZE = N_COLS
INV_N = 1.0 / N_COLS

# Each CTA processes multiple rows to amortize launch overhead and keep
# the SM busy – this is the key refinement over the one‑row‑per‑CTA parent.
ROWS_PER_PROG = 8          # 8 rows per block, grid = N_ROWS // 8

@triton.jit
def welford_kernel(
    x_ptr,
    out_ptr,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INV_N: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    first_row = pid * ROWS_PER_PROG

    # Each program loops over all its assigned rows, re‑using the same registers.
    for r in tl.static_range(0, ROWS_PER_PROG):
        row = first_row + r
        row_start = row * n_cols

        # Vectorized, coalesced load of the whole row (no mask needed)
        offsets = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
        x_ptrs = x_ptr + row_start + offsets
        x = tl.load(x_ptrs, eviction_policy='evict_first')

        # Single‑pass sum & sum‑of‑squares in fp32
        s  = tl.sum(x, axis=0).to(tl.float32)
        sq = tl.sum(x * x, axis=0).to(tl.float32)

        mean = s * INV_N
        var  = (sq * INV_N) - mean * mean   # population variance

        # Store results – output layout [2, n_rows]
        tl.store(out_ptr + 0 * n_rows + row, mean)
        tl.store(out_ptr + 1 * n_rows + row, var)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32
    assert x.shape == (N_ROWS, N_COLS), f"Expected ({N_ROWS}, {N_COLS}) but got {x.shape}"

    out = torch.empty((2, N_ROWS), dtype=torch.float32, device=x.device)

    grid = (N_ROWS // ROWS_PER_PROG,)
    welford_kernel[grid](
        x, out,
        n_rows=N_ROWS,
        n_cols=N_COLS,
        BLOCK_SIZE=BLOCK_SIZE,
        INV_N=INV_N,
        ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=8,          # 8 warps gave best bandwidth in the parent
        num_stages=2,         # low stages keep register pressure low
    )
    return out