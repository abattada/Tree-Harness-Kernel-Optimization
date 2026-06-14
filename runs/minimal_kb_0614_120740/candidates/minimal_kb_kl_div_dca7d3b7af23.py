import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 1: per-row KL divergence (sum over columns of q*(log(q)-log_p))
# Specialized with constexpr rows/cols for loop unrolling and mask elimination.
# ---------------------------------------------------------------------------
@triton.jit
def kl_div_row_kernel(
    log_p_ptr,          # f32 [rows, cols]
    q_ptr,              # f32 [rows, cols]
    row_sum_ptr,        # f32 [rows]  (output)
    BLOCK_SIZE: tl.constexpr,
):
    # Shapes are constexpr; assert cols divisible by BLOCK_SIZE for mask removal
    rows: tl.constexpr = 8192
    cols: tl.constexpr = 8192
    tl.static_assert(cols % BLOCK_SIZE == 0, "cols not divisible by BLOCK_SIZE")
    tl.static_assert(rows == 8192, "rows must be 8192")

    pid = tl.program_id(0)
    row_start = pid * cols

    acc = tl.zeros([], dtype=tl.float32)

    # Fully unrolled loop over columns (2 iterations with BLOCK_SIZE=4096)
    for col_start in range(0, cols, BLOCK_SIZE):
        offsets = col_start + tl.arange(0, BLOCK_SIZE)
        # No mask needed because offsets < cols always true due to divisibility
        q_vals = tl.load(
            q_ptr + row_start + offsets,
            eviction_policy='evict_first',
        )
        log_p_vals = tl.load(
            log_p_ptr + row_start + offsets,
            eviction_policy='evict_first',
        )
        # Safe term: q * (log(q) - log_p), with 0 when q==0
        term = tl.where(q_vals > 0, q_vals * (tl.log(q_vals) - log_p_vals), 0.0)
        acc += tl.sum(term)

    tl.store(row_sum_ptr + pid, acc)


# ---------------------------------------------------------------------------
# Kernel 2: reduction of row sums to a single scalar (mean over rows)
# Also constexpr-specialized for single-block reduction.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_kernel(
    row_sum_ptr,   # f32 [rows]
    scalar_ptr,    # f32 [1]
):
    rows: tl.constexpr = 8192
    BLOCK_SIZE: tl.constexpr = 8192
    tl.static_assert(rows % BLOCK_SIZE == 0, "rows not divisible by BLOCK_SIZE")

    total = tl.zeros([], dtype=tl.float32)
    # Single iteration, no mask needed
    offsets = tl.arange(0, BLOCK_SIZE)
    vals = tl.load(row_sum_ptr + offsets, eviction_policy='evict_first')
    total += tl.sum(vals)
    tl.store(scalar_ptr, total / rows)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def triton_run(log_p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Compute Kullback–Leibler divergence with reduction='batchmean'.
    Input shapes are both (8192, 8192).
    Returns a scalar tensor.
    """
    rows, cols = log_p.shape
    assert q.shape == (rows, cols)
    assert rows == 8192 and cols == 8192, "only 8192x8192 supported"

    # Allocate intermediate row sums and final scalar output
    row_sum = torch.empty(rows, dtype=torch.float32, device=log_p.device)
    scalar_out = torch.empty(1, dtype=torch.float32, device=log_p.device)

    # Row kernel: BLOCK_SIZE=4096 for 2 iterations; 8 warps to hide latency
    grid_row = (rows,)
    kl_div_row_kernel[grid_row](
        log_p, q, row_sum,
        BLOCK_SIZE=4096,
        num_warps=8,
    )

    # Reduction kernel: single block, 4 warps for quick reduction
    grid_reduce = (1,)
    reduce_kernel[grid_reduce](
        row_sum, scalar_out,
        num_warps=4,
    )

    return scalar_out.view(())