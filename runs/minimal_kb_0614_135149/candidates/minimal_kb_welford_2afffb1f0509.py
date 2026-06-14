import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Fixed dimensions: 8192 rows x 4096 cols. We exploit compile-time constants
# to tell the compiler the shape is fixed, eliminating all boundary checks.
# We also hint that row pointers are 128‑byte aligned so Triton can use the
# widest possible vector loads (e.g. LDG.128) and approach peak memory BW.
# ---------------------------------------------------------------------------
N_ROWS = 8192
N_COLS = 4096
BLOCK_SIZE = N_COLS
INV_N = 1.0 / N_COLS          # compile-time reciprocal – no division in kernel

@triton.jit
def welford_kernel(
    x_ptr,                     # [N_ROWS, N_COLS] input, float32
    out_ptr,                   # [2, N_ROWS] output  (row0=mean, row1=var)
    n_rows: tl.constexpr,      # 8192
    n_cols: tl.constexpr,      # 4096
    BLOCK_SIZE: tl.constexpr,
    INV_N: tl.constexpr,
):
    pid = tl.program_id(0)                 # row index
    row_offset = pid * n_cols

    # ------------------------------------------------------------------
    # Hint: the start address of every row is 128‑byte aligned (16 KiB row
    # size). This allows the compiler to emit wide memory transactions.
    # ------------------------------------------------------------------
    row_offset = tl.multiple_of(row_offset, 128)

    # Load the whole row contiguously – no mask required because n_cols
    # divides BLOCK_SIZE exactly. evict_first keeps the one‑shot data from
    # polluting the L2/L1.
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
    x_ptrs = x_ptr + row_offset + offsets
    x = tl.load(x_ptrs, eviction_policy='evict_first')

    # Single‑pass sum and sum of squares, all in fp32.
    s  = tl.sum(x, axis=0).to(tl.float32)
    sq = tl.sum(x * x, axis=0).to(tl.float32)

    # Population mean & variance using the pre‑computed inverse.
    mean = s * INV_N
    var = (sq * INV_N) - mean * mean

    # Output layout: [2, N_ROWS], column‑major with stride = n_rows.
    out_stride = n_rows
    tl.store(out_ptr + 0 * out_stride + pid, mean)
    tl.store(out_ptr + 1 * out_stride + pid, var)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    Per‑row mean and population variance of a float32 (8192, 4096) tensor.

    Returns: out tensor of shape (2, 8192), dtype float32.
    """
    assert x.is_cuda and x.dtype == torch.float32
    assert x.shape == (N_ROWS, N_COLS), f"Expected ({N_ROWS}, {N_COLS}), got {x.shape}"

    out = torch.empty((2, N_ROWS), dtype=torch.float32, device=x.device)

    grid = (N_ROWS,)
    welford_kernel[grid](
        x,
        out,
        n_rows=N_ROWS,
        n_cols=N_COLS,
        BLOCK_SIZE=BLOCK_SIZE,
        INV_N=INV_N,
        num_warps=8,          # good balance for memory‑parallelism on this chip
        num_stages=2,         # low stages keeps register pressure minimal
    )
    return out