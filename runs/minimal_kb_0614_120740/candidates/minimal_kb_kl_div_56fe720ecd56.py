import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 1: per-row KL divergence sum
#   Each program processes one full row (cols = 8192).
# ---------------------------------------------------------------------------
@triton.jit
def row_kl_kernel(
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
    # Single block iteration since BLOCK_SIZE == cols
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
        # term = q * (log(q) - log_p), safe when q == 0
        term = tl.where(q_vals > 0, q_vals * (tl.log(q_vals) - log_p_vals), 0.0)
        acc += tl.sum(term)

    tl.store(row_sum_ptr + pid, acc)


# ---------------------------------------------------------------------------
# Kernel 2: reduce per-row sums to a single scalar, then divide by rows
#   Process all rows in one iteration (BLOCK_SIZE = rows).
# ---------------------------------------------------------------------------
@triton.jit
def reduce_row_sums_kernel(
    row_sum_ptr,        # f32 [rows]
    scalar_ptr,         # f32 [1]
    rows: tl.constexpr,
):
    # Single program, single warp – reduce full vector of size rows (8192)
    offsets = tl.arange(0, rows)  # rows is constexpr, so this is compile-time constant
    mask = offsets < rows
    vals = tl.load(
        row_sum_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy='evict_first',
    )
    total = tl.sum(vals)
    tl.store(scalar_ptr, total / rows)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def triton_run(log_p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    KL divergence with reduction='batchmean'.
    Both tensors are (8192, 8192), float32.
    Returns a scalar (0-D) tensor.
    """
    assert log_p.shape == q.shape
    rows, cols = log_p.shape

    # Allocate intermediate row sums and final scalar
    row_sum = torch.empty(rows, dtype=torch.float32, device=log_p.device)
    scalar_out = torch.empty(1, dtype=torch.float32, device=log_p.device)

    # Launch row kernel: one program per row, BLOCK_SIZE = cols = 8192
    # Use 4 warps – good balance for memory‑bandwidth‑bound workloads
    BLOCK_SIZE_ROW = 8192
    grid_row = (rows,)
    row_kl_kernel[grid_row](
        log_p, q, row_sum,
        rows, cols, BLOCK_SIZE_ROW,
        num_warps=4,
    )

    # Launch reduction kernel: single program, processes all rows at once
    # One warp is enough for a single warp‑level reduction of 8192 elements
    reduce_row_sums_kernel[(1,)](
        row_sum, scalar_out,
        rows,
        num_warps=1,
    )

    # Return scalar (0‑D tensor)
    return scalar_out[0]