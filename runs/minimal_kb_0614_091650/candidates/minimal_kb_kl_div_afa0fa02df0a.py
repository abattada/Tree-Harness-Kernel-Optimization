import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 1: per-row KL divergence (batchmean) – each program handles one row.
# Loads q and log_p in tiles, accumulates the row sum, writes partial result.
# Uses cache eviction hints to minimize L2 pollution and memory annotations
# to help the compiler with vectorization.
# ---------------------------------------------------------------------------
@triton.jit
def kl_div_row_kernel(
    log_p_ptr,          # f32 [rows, cols]
    q_ptr,              # f32 [rows, cols]
    row_sum_ptr,        # f32 [rows]   – per-row partial sums
    rows: tl.constexpr,
    cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,          # tile along cols
):
    pid = tl.program_id(0)                     # each program = one row
    row_start = pid * cols

    acc = tl.zeros([], dtype=tl.float32)
    # Loop over tiles along the column dimension
    for col_start in range(0, cols, BLOCK_SIZE):
        offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < cols

        # Load q and log_p with hints for coalesced, aligned access
        q_vals = tl.load(
            q_ptr + row_start + offsets,
            mask,
            other=0.0,
            cache_modifier='.cg',                 # keep in L1/L2, default
            eviction_policy='evict_first'          # streaming, don't pollute L2
        )
        log_p_vals = tl.load(
            log_p_ptr + row_start + offsets,
            mask,
            other=0.0,
            eviction_policy='evict_first'
        )

        # term = q * (log(q) - log_p); avoid log(0)*0 -> NaN
        term = tl.where(q_vals > 0.0,
                        q_vals * (tl.log(q_vals) - log_p_vals),
                        0.0)
        acc += tl.sum(term)

    # Store the row sum; prefer evict_last because it will be read soon
    tl.store(
        row_sum_ptr + pid,
        acc,
        eviction_policy='evict_last'
    )


# ---------------------------------------------------------------------------
# Kernel 2: reduce the per-row sums into a scalar (batchmean division).
# ---------------------------------------------------------------------------
@triton.jit
def reduce_row_sums_kernel(
    row_sum_ptr,        # f32 [rows]
    scalar_ptr,         # f32 [1]
    rows: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    total = tl.zeros([], dtype=tl.float32)
    for start in range(0, rows, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < rows
        vals = tl.load(
            row_sum_ptr + offsets,
            mask,
            other=0.0,
            eviction_policy='evict_first'          # read-once, evict quickly
        )
        total += tl.sum(vals)

    # All threads have the same total; store by thread 0 (all write same value)
    tl.store(scalar_ptr, total / rows)


# ---------------------------------------------------------------------------
# Public entry point: allocates outputs, launches both kernels, returns scalar.
# ---------------------------------------------------------------------------
def triton_run(log_p, q) -> torch.Tensor:
    # Input shapes (tensors on GPU)
    rows, cols = log_p.shape
    assert q.shape == (rows, cols)

    row_sum = torch.empty(rows, dtype=torch.float32, device=log_p.device)
    scalar_out = torch.empty(1, dtype=torch.float32, device=log_p.device)

    BLOCK_SIZE = 1024  # divides 8192 evenly, masks are still safe

    grid_row = (rows,)
    kl_div_row_kernel[grid_row](
        log_p, q, row_sum,
        rows, cols, BLOCK_SIZE,
        num_warps=8,
    )

    grid_red = (1,)
    reduce_row_sums_kernel[grid_red](
        row_sum, scalar_out,
        rows, BLOCK_SIZE,
        num_warps=4,               # small reduction, fewer warps enough
    )

    return scalar_out.squeeze(0)

# ---------------------------------------------------------------------------
# End of module
# ---------------------------------------------------------------------------