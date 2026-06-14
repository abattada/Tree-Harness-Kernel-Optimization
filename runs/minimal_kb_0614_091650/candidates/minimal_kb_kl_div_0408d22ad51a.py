import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 1: per-row KL divergence, each program handles one row.
# We use a larger BLOCK_SIZE (2048) to reduce loop iterations,
# add cache eviction hints and alignment hints for better memory throughput.
# ---------------------------------------------------------------------------
@triton.jit
def kl_div_row_kernel(
    log_p_ptr,          # f32 [rows, cols]
    q_ptr,              # f32 [rows, cols]
    row_sum_ptr,        # f32 [rows]  (output)
    rows: tl.constexpr,
    cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * cols

    acc = tl.zeros([], dtype=tl.float32)
    # Loop over columns in BLOCK_SIZE-sized chunks
    for col_start in range(0, cols, BLOCK_SIZE):
        offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < cols
        # Load q and log_p with evict_first (streaming access)
        q_vals = tl.load(
            q_ptr + row_start + offsets,
            mask=mask,
            other=0.0,
            eviction_policy='evict_first',
        )
        log_p_vals = tl.load(
            log_p_ptr + row_start + offsets,
            mask=mask,
            other=0.0,
            eviction_policy='evict_first',
        )
        # term = q * (log(q) - log_p)  ; avoid 0 * (-inf) = NaN
        term = tl.where(q_vals > 0, q_vals * (tl.log(q_vals) - log_p_vals), 0.0)
        acc += tl.sum(term)

    tl.store(row_sum_ptr + pid, acc)


# ---------------------------------------------------------------------------
# Kernel 2: reduce per-row sums to a scalar (mean over rows).
# We use a single warp (num_warps=1) to reduce the 8192 elements in two
# iterations (BLOCK_SIZE=4096) for low overhead.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_kernel(
    row_sum_ptr,   # f32 [rows]
    scalar_ptr,    # f32 [1]
    rows: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    total = tl.zeros([], dtype=tl.float32)
    for start in range(0, rows, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < rows
        vals = tl.load(row_sum_ptr + offsets, mask=mask, other=0.0)
        total += tl.sum(vals)
    # Store the mean
    tl.store(scalar_ptr, total / rows)


def triton_run(log_p, q) -> torch.Tensor:
    assert log_p.shape == q.shape
    rows, cols = log_p.shape

    # Allocate intermediate row sums and final scalar
    row_sum = torch.empty(rows, dtype=torch.float32, device=log_p.device)
    scalar_out = torch.empty(1, dtype=torch.float32, device=log_p.device)

    BLOCK_SIZE = 2048  # increased from 1024 to reduce loop iterations
    grid_row = (rows,)
    kl_div_row_kernel[grid_row](
        log_p, q, row_sum,
        rows, cols, BLOCK_SIZE,
        num_warps=8,   # keep – good balance for RTX 5090
    )

    # Reduction kernel: larger block and single warp
    BLOCK_RED = 4096
    grid_red = (1,)
    reduce_kernel[grid_red](
        row_sum, scalar_out,
        rows, BLOCK_RED,
        num_warps=1,   # single warp enough for 8192 elements via loop
    )

    return scalar_out.squeeze(0)