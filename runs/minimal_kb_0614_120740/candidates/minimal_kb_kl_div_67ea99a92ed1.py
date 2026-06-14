import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 1: per-row KL divergence, each program handles one row.
# We use a block size of 2048 to reduce loop iterations, cache eviction hints,
# and alignment hints for better memory throughput.
# ---------------------------------------------------------------------------
@triton.jit
def kl_div_row_kernel(
    log_p_ptr,          # f32 [rows, cols]
    q_ptr,              # f32 [rows, cols]
    row_sum_ptr,        # f32 [rows]  (output per row sum)
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
# A single block sums the row sums, then divides by rows.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_kernel(
    row_sum_ptr,   # f32 [rows]
    scalar_ptr,    # f32 [1]  (we store the final scalar)
    rows: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    total = tl.zeros([], dtype=tl.float32)
    for start in range(0, rows, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < rows
        vals = tl.load(
            row_sum_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy='evict_first',
        )
        total += tl.sum(vals)
    # Store the mean (total / rows) into the scalar output
    tl.store(scalar_ptr, total / rows)


# ---------------------------------------------------------------------------
# Public entry point: returns 0‑d scalar tensor with KL divergence (batchmean)
# ---------------------------------------------------------------------------
def triton_run(log_p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Compute Kullback–Leibler divergence with reduction='batchmean'.
    Input shapes are both (8192, 8192).
    Returns a 0‑d tensor (scalar) on the same device.
    """
    assert log_p.shape == q.shape
    rows, cols = log_p.shape

    # Allocate intermediate row sums and final scalar
    row_sum = torch.empty(rows, dtype=torch.float32, device=log_p.device)
    scalar_out = torch.empty((), dtype=torch.float32, device=log_p.device)  # 0‑d

    # Tunable block sizes – 2048 works well for 8192 columns
    BLOCK_SIZE = 2048
    grid_row = (rows,)
    kl_div_row_kernel[grid_row](
        log_p, q, row_sum,
        rows, cols,
        BLOCK_SIZE,
        num_warps=4,       # moderate warps for 2048 elements
        num_stages=3,      # default stages
    )

    # Reduce row sums to a scalar: one block with a single warp
    # Use a block size that covers all rows (or loop)
    BLOCK_REDUCE = 4096  # enough to cover 8192 rows in two iterations
    grid_reduce = (1,)
    reduce_kernel[grid_reduce](
        row_sum, scalar_out,
        rows,
        BLOCK_REDUCE,
        num_warps=1,
        num_stages=2,
    )

    return scalar_out