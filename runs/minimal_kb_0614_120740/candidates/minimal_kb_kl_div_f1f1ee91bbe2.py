import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 1: per-row KL divergence.
# For each row i, compute sum_j q[i,j] * (log(q[i,j]) - log_p[i,j]).
# Handles q == 0 safely (0 * log(0) = 0).
# BLOCK_SIZE should divide cols evenly for mask-free access (cols = 8192).
# ---------------------------------------------------------------------------
@triton.jit
def row_kl_kernel(
    log_p_ptr,          # [rows, cols]
    q_ptr,              # [rows, cols]
    row_sum_ptr,        # [rows] output: row-local KL sum
    rows: tl.constexpr,
    cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * cols

    acc = tl.zeros([], dtype=tl.float32)

    # Loop over columns in BLOCK_SIZE‑sized chunks
    for col_start in range(0, cols, BLOCK_SIZE):
        offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < cols

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

        # term = q * (log(q) - log_p), with safe handling for q == 0
        term = tl.where(q_vals > 0, q_vals * (tl.log(q_vals) - log_p_vals), 0.0)
        acc += tl.sum(term)

    tl.store(row_sum_ptr + pid, acc)


# ---------------------------------------------------------------------------
# Kernel 2: reduction of per‑row sums to a single scalar (mean over rows).
# Uses one warp to minimise overhead.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_row_sums_kernel(
    row_sum_ptr,        # [rows]
    scalar_ptr,         # [1] output
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

    # All threads hold the same total; store the mean.
    tl.store(scalar_ptr, total / rows)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def triton_run(log_p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Compute Kullback–Leibler divergence with reduction='batchmean'.
    Input shapes must be (rows, cols), both f32.
    Returns a scalar tensor (shape [1]) on the same device.
    """
    rows, cols = log_p.shape
    assert q.shape == (rows, cols)

    # Allocate intermediate per‑row sums and final scalar
    row_sum = torch.empty(rows, dtype=torch.float32, device=log_p.device)
    scalar_out = torch.empty(1, dtype=torch.float32, device=log_p.device)

    # ---------- Kernel 1: row‑wise KL ----------
    BLOCK_SIZE_ROW = 2048          # divides cols (8192) exactly -> mask mostly static
    grid_row = (rows,)
    row_kl_kernel[grid_row](
        log_p, q, row_sum,
        rows, cols, BLOCK_SIZE_ROW,
        num_warps=4,               # good balance for memory‑bound row loop
    )

    # ---------- Kernel 2: reduce rows -> scalar ----------
    BLOCK_SIZE_REDUCE = 4096       # 8192 rows covered in 2 iterations
    grid_reduce = (1,)             # single block
    reduce_row_sums_kernel[grid_reduce](
        row_sum, scalar_out,
        rows, BLOCK_SIZE_REDUCE,
        num_warps=1,               # single warp is sufficient
    )

    return scalar_out