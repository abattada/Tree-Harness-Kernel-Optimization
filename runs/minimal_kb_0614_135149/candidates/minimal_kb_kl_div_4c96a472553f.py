import torch
import triton
import triton.language as tl


@triton.jit
def kl_persistent_kernel(
    log_p_ptr,
    q_ptr,
    scalar_ptr,
    rows: tl.constexpr,
    cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Persistent kernel: each program processes multiple rows via a grid-stride
    loop and atomically accumulates the total KL divergence sum.
    Intermediate row sums are omitted — final combination is done with atomic_add.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    total = tl.zeros([], dtype=tl.float32)

    # Grid-stride loop over rows
    for row in range(pid, rows, num_programs):
        row_start = row * cols
        # Loop over columns of one row
        for col_start in range(0, cols, BLOCK_SIZE):
            offsets = tl.max_contiguous(
                tl.multiple_of(col_start + tl.arange(0, BLOCK_SIZE), BLOCK_SIZE),
                BLOCK_SIZE,
            )
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
            # term = q * (log(q) - log_p)  ; safe for q == 0
            term = tl.where(q_vals > 0.0, q_vals * (tl.log(q_vals) - log_p_vals), 0.0)
            total += tl.sum(term)

    # Contribute per-program partial sum to global scalar
    tl.atomic_add(scalar_ptr, total)


def triton_run(log_p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence with reduction='batchmean'.
    Inputs: log_p, q  — both f32[8192, 8192]
    Returns: f32[] scalar
    """
    assert log_p.shape == q.shape, "Input shapes must match"
    assert log_p.dtype == q.dtype == torch.float32, "Inputs must be float32"
    rows, cols = log_p.shape

    # Single output scalar, pre-zeroed for atomic_add accumulation
    scalar_out = torch.zeros(1, dtype=torch.float32, device=log_p.device)

    # Launch persistent kernel: fewer programs than rows to reduce atomic contention
    # and amortize launch overhead. 1024 programs → ~8 rows each on 8192.
    grid = (min(rows, 1024),)
    BLOCK_SIZE = 2048  # larger block → fewer column iterations per row

    kl_persistent_kernel[grid](
        log_p, q, scalar_out,
        rows=rows, cols=cols, BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8, num_stages=2,
    )

    # Convert total sum to batchmean (sum / rows)
    scalar_out /= rows

    return scalar_out.squeeze()