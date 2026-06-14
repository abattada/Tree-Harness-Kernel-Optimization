import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: grid‑stride row KL calculator
#   Each program loops over multiple rows (stride = GRID_SIZE) and
#   accumulates the total KL sum → writes one partial sum.
# ---------------------------------------------------------------------------
@triton.jit
def kl_grid_stride_kernel(
    log_p_ptr,          # f32 [rows, cols]
    q_ptr,              # f32 [rows, cols]
    partial_ptr,        # f32 [GRID_SIZE]  – one partial per program
    rows: tl.constexpr,
    cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GRID_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total = tl.zeros([], dtype=tl.float32)

    # Each program handles rows: pid, pid+GRID_SIZE, pid+2*GRID_SIZE, …
    for row in range(pid, rows, GRID_SIZE):
        base = row * cols
        for col_start in range(0, cols, BLOCK_SIZE):
            offs = col_start + tl.arange(0, BLOCK_SIZE)
            # cols = 8192, BLOCK_SIZE = 2048 → exact division, no mask needed.
            q_vals = tl.load(q_ptr + base + offs,
                             eviction_policy='evict_first')
            log_p_vals = tl.load(log_p_ptr + base + offs,
                                 eviction_policy='evict_first')
            # term = q * (log(q) - log_p), safely handling q == 0
            term = tl.where(q_vals > 0.0,
                            q_vals * (tl.log(q_vals) - log_p_vals), 0.0)
            total += tl.sum(term)

    tl.store(partial_ptr + pid, total)


# ---------------------------------------------------------------------------
# Kernel 2: reduce the partial sums to a scalar and divide by rows (batchmean)
# ---------------------------------------------------------------------------
@triton.jit
def reduce_partial_kernel(
    partial_ptr,        # f32 [GRID_SIZE]
    out_ptr,            # f32 [1]
    rows: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    BLOCK_REDUCE: tl.constexpr,
):
    total = tl.zeros([], dtype=tl.float32)
    for start in range(0, GRID_SIZE, BLOCK_REDUCE):
        offs = start + tl.arange(0, BLOCK_REDUCE)
        mask = offs < GRID_SIZE
        vals = tl.load(partial_ptr + offs, mask=mask, other=0.0)
        total += tl.sum(vals)

    # batchmean = total_sum / rows
    tl.store(out_ptr, total / rows)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def triton_run(log_p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    KL divergence with batchmean reduction: sum(q*(log q - log_p)) / batch_size.
    Input shapes are (8192, 8192) float32; returns scalar tensor.
    """
    rows, cols = log_p.shape
    assert log_p.shape == q.shape, "Input shapes must match"
    assert log_p.dtype == torch.float32, "Inputs must be float32"

    # Tuned constants for 8192×8192
    BLOCK_SIZE = 2048      # divides 8192 evenly → 4 column passes
    GRID_SIZE = 256        # number of programs; each processes 8192/256 = 32 rows
    BLOCK_REDUCE = 256      # matches GRID_SIZE → one pass in the reduction

    partial = torch.empty(GRID_SIZE, dtype=torch.float32, device=log_p.device)
    scalar_out = torch.empty(1, dtype=torch.float32, device=log_p.device)

    # Row‑processing kernel – 256 programs, stride over rows
    kl_grid_stride_kernel[(GRID_SIZE,)](
        log_p, q, partial,
        rows=rows, cols=cols,
        BLOCK_SIZE=BLOCK_SIZE,
        GRID_SIZE=GRID_SIZE,
        num_warps=16, num_stages=2,
    )

    # Reduction kernel – single program summing the partials
    reduce_partial_kernel[(1,)](
        partial, scalar_out,
        rows=rows,
        GRID_SIZE=GRID_SIZE,
        BLOCK_REDUCE=BLOCK_REDUCE,
        num_warps=1,
    )

    return scalar_out.squeeze()