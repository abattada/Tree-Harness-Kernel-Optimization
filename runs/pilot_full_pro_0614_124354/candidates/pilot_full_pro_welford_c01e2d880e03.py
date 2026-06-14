import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    Compute per-row population mean and variance using a single-pass sum-of-squares approach.
    Multirow strategy: each program processes ROWS_PER_PROG rows to amortize launch overhead
    and improve SM utilization.
    """
    N_ROWS, N_COLS = x.shape         # (8192, 4096)
    out = torch.empty(2, N_ROWS, dtype=x.dtype, device=x.device)

    # Tuned knobs for RTX 5090 (Blackwell)
    BLOCK_SIZE = 256          # tile size along columns; must divide N_COLS exactly
    NUM_WARPS = 8             # 8 warps → 256 threads, good occupancy with multirow
    NUM_STAGES = 2           # pipelining for streaming loads
    ROWS_PER_PROG = 4        # each threadblock handles 4 rows

    assert N_COLS % BLOCK_SIZE == 0, "BLOCK_SIZE must divide N_COLS"
    grid = (triton.cdiv(N_ROWS, ROWS_PER_PROG),)
    welford_kernel[grid](
        x, out,
        x.stride(0),
        N_ROWS, N_COLS,
        BLOCK_SIZE=BLOCK_SIZE,
        ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return out


@triton.jit
def welford_kernel(x_ptr, out_ptr, stride_row,
                   N_ROWS: tl.constexpr, N_COLS: tl.constexpr,
                   BLOCK_SIZE: tl.constexpr, ROWS_PER_PROG: tl.constexpr):
    """
    Each program computes mean and variance for ROWS_PER_PROG contiguous rows.
    The column reduction uses a vectorized tile loop with no boundary masks.
    """
    tl.static_assert(N_COLS % BLOCK_SIZE == 0,
                     "N_COLS must be divisible by BLOCK_SIZE for boundary‑free loop")

    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROG

    # Process each row in the program's assigned chunk
    for r in tl.static_range(0, ROWS_PER_PROG):
        row_idx = row_start + r
        if row_idx < N_ROWS:
            row_base = x_ptr + row_idx * stride_row
            row_base = tl.multiple_of(row_base, BLOCK_SIZE)

            acc_sum = tl.zeros([], dtype=tl.float32)
            acc_sum_sq = tl.zeros([], dtype=tl.float32)

            # Fully unrolled column tile reduction
            for col_start in tl.static_range(0, N_COLS, BLOCK_SIZE):
                offsets = col_start + tl.arange(0, BLOCK_SIZE)
                vals = tl.load(row_base + offsets, cache_modifier='.cg')
                acc_sum += tl.sum(vals, axis=0)
                acc_sum_sq += tl.sum(vals * vals, axis=0)

            N = tl.cast(N_COLS, tl.float32)
            mean = acc_sum / N
            var = acc_sum_sq / N - mean * mean  # population variance

            # Store results: [mean, var] for row_idx
            tl.store(out_ptr + row_idx, mean)
            tl.store(out_ptr + N_ROWS + row_idx, var)