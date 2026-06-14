import torch
import triton
import triton.language as tl


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    Compute per‑row population mean and variance using a single‑pass
    sum‑of‑squares approach.  The block size has been tuned for the
    (8192, 4096) problem size to balance loop overhead and register
    pressure, targeting the RTX 5090 memory bandwidth.
    """
    N_ROWS, N_COLS = x.shape
    out = torch.empty(2, N_ROWS, dtype=x.dtype, device=x.device)

    # Tuned configuration (chosen after sweeping powers of two)
    BLOCK_SIZE = 1024        # divides N_COLS exactly (4096 % 1024 == 0)
    NUM_WARPS = 4
    NUM_STAGES = 2

    assert N_COLS % BLOCK_SIZE == 0, "BLOCK_SIZE must divide N_COLS"

    grid = (N_ROWS,)
    welford_kernel[grid](
        x, out,
        x.stride(0),
        N_ROWS=N_ROWS, N_COLS=N_COLS,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return out


@triton.jit
def welford_kernel(x_ptr, out_ptr, stride_row,
                   N_ROWS: tl.constexpr, N_COLS: tl.constexpr,
                   BLOCK_SIZE: tl.constexpr):
    """
    One program per row.  Each row is processed in BLOCK_SIZE‑wide
    vectorised tiles.  A warp‑local cross‑thread reduction on every tile
    updates two scalar accumulators; the final mean and population
    variance are computed once per row.
    """
    tl.static_assert(N_COLS % BLOCK_SIZE == 0,
                     "N_COLS must be divisible by BLOCK_SIZE for mask‑free loop")

    pid = tl.program_id(0)                # row index
    row_base = x_ptr + pid * stride_row
    row_base = tl.multiple_of(row_base, BLOCK_SIZE)

    # Accumulators (fp32 is mandatory for numerical stability of sums)
    acc_sum   = tl.zeros([], dtype=tl.float32)
    acc_sum_sq = tl.zeros([], dtype=tl.float32)

    # Fully unrolled loop over column tiles – no boundary mask needed
    for col_start in tl.static_range(0, N_COLS, BLOCK_SIZE):
        offsets = col_start + tl.arange(0, BLOCK_SIZE)
        vals = tl.load(row_base + offsets, cache_modifier='.cg')
        acc_sum   += tl.sum(vals, axis=0)
        acc_sum_sq += tl.sum(vals * vals, axis=0)

    N = tl.cast(N_COLS, tl.float32)
    mean = acc_sum / N
    var  = acc_sum_sq / N - mean * mean      # population variance (unbiased=False)

    # Store into contiguous (2, N_ROWS) output
    tl.store(out_ptr + pid, mean)
    tl.store(out_ptr + N_ROWS + pid, var)