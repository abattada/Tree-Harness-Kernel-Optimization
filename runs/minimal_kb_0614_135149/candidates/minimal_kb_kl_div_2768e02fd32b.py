import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 1: per-row KL divergence sum – one program per row.
# Processes the entire row (8192 floats) in a single block with vectorized
# loads, explicit contiguity hints and cache eviction hints.
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

    # All offsets cover one full row, contiguous -> max_contiguous = cols
    offsets = tl.arange(0, BLOCK_SIZE)

    q_vals = tl.load(
        q_ptr + row_start + offsets,
        mask=None,                  # no mask because BLOCK_SIZE == cols
        other=0.0,
        eviction_policy='evict_first',
        max_contiguous=cols,        # hint for vectorized load
    )
    log_p_vals = tl.load(
        log_p_ptr + row_start + offsets,
        mask=None,
        other=0.0,
        eviction_policy='evict_first',
        max_contiguous=cols,
    )

    # term = q * (log(q) - log_p)  ; q == 0 handled safely
    term = tl.where(
        q_vals > 0.0,
        q_vals * (tl.log(q_vals) - log_p_vals),
        0.0,
    )

    # Full-row reduction (tree sum in shared memory)
    acc = tl.sum(term)
    tl.store(row_sum_ptr + pid, acc)


# ---------------------------------------------------------------------------
# Kernel 2: reduce per-row sums to a scalar (batchmean).
# ---------------------------------------------------------------------------
@triton.jit
def reduce_scalar_kernel(
    row_sum_ptr,        # f32 [rows]
    scalar_ptr,         # f32 [1]
    rows: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    vals = tl.load(
        row_sum_ptr + offsets,
        mask=None,                  # BLOCK_SIZE == rows
        other=0.0,
        eviction_policy='evict_first',
        max_contiguous=rows,
    )
    total = tl.sum(vals)
    mean = total / rows
    tl.store(scalar_ptr, mean)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def triton_run(log_p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    KL divergence with batchmean reduction:
        sum(q * (log q - log_p)) / batch_size.
    Inputs: both [8192, 8192], float32. Returns scalar float32 tensor.
    """
    rows, cols = log_p.shape
    assert log_p.shape == q.shape, "Input shapes must match"
    assert log_p.dtype == torch.float32, "Inputs must be float32"

    # Allocate intermediate row sums and output scalar
    row_sum = torch.empty(rows, dtype=torch.float32, device=log_p.device)
    scalar_out = torch.empty(1, dtype=torch.float32, device=log_p.device)

    # Shapes as compile‑time constants for specialization
    ROWS = rows   # 8192
    COLS = cols   # 8192

    # Tuned constants: one block covers one whole row / whole reduction array
    BLOCK_COLS = 8192
    BLOCK_ROWS = 8192

    # Row kernel: one program per row, loads entire row at once
    row_kl_kernel[(ROWS,)](
        log_p, q, row_sum,
        rows=ROWS, cols=COLS, BLOCK_SIZE=BLOCK_COLS,
        num_warps=16,
    )

    # Reduction kernel: single program reduces all row sums in one go
    reduce_scalar_kernel[(1,)](
        row_sum, scalar_out,
        rows=ROWS, BLOCK_SIZE=BLOCK_ROWS,
        num_warps=2,
    )

    return scalar_out.squeeze()