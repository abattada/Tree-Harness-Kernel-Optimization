import torch
import triton
import triton.language as tl


@triton.jit
def kl_row_kernel(
    log_p_ptr,         # f32[rows, cols]
    q_ptr,             # f32[rows, cols]
    row_sum_ptr,       # f32[rows]
    cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Process one whole row at a time. BLOCK_SIZE must equal cols.
    """
    pid = tl.program_id(0)
    row_start = pid * cols

    offsets = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
    q_vals = tl.load(q_ptr + row_start + offsets, eviction_policy='evict_first')
    log_p_vals = tl.load(log_p_ptr + row_start + offsets, eviction_policy='evict_first')

    # safe log(q) for q>0, 0 otherwise
    term = tl.where(q_vals > 0, q_vals * (tl.log(q_vals) - log_p_vals), 0.0)
    acc = tl.sum(term)
    tl.store(row_sum_ptr + pid, acc)


@triton.jit
def reduce_kernel(
    row_sum_ptr,       # f32[rows]
    scalar_ptr,        # f32[1]
    rows: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Single-block reduction: sum all row sums and divide by rows.
    BLOCK_SIZE must equal rows.
    """
    offsets = tl.arange(0, BLOCK_SIZE)
    vals = tl.load(row_sum_ptr + offsets, eviction_policy='evict_first')
    total = tl.sum(vals)
    tl.store(scalar_ptr, total / rows)


def triton_run(log_p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence with reduction='batchmean'.
    Inputs: log_p, q  — both f32[8192, 8192]
    Returns: f32[] scalar
    """
    assert log_p.shape == q.shape, "Input shapes must match"
    assert log_p.dtype == q.dtype == torch.float32, "Inputs must be float32"
    rows, cols = log_p.shape

    row_sum = torch.empty(rows, dtype=torch.float32, device=log_p.device)
    scalar_out = torch.empty(1, dtype=torch.float32, device=log_p.device)

    # Each program loads a complete row (cols == 8192)
    kl_row_kernel[(rows,)](
        log_p, q, row_sum,
        cols=cols, BLOCK_SIZE=cols,
        num_warps=8, num_stages=2,
    )

    # Single block reduces all row sums
    reduce_kernel[(1,)](
        row_sum, scalar_out,
        rows=rows, BLOCK_SIZE=rows,
        num_warps=8, num_stages=2,
    )

    return scalar_out.squeeze()