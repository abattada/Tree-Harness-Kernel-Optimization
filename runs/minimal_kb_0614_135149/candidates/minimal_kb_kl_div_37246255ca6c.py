import torch
import triton
import triton.language as tl


@triton.jit
def kl_chunk_kernel(
    log_p_ptr,          # f32 [rows, cols]
    q_ptr,              # f32 [rows, cols]
    partial_sum_ptr,    # f32 [num_chunks]
    rows: tl.constexpr,
    cols: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Each program processes a contiguous chunk of rows (up to ROWS_PER_PROG),
    summing the per-row KL divergence and writing a single partial total.
    """
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROG
    row_end = tl.minimum(row_start + ROWS_PER_PROG, rows)

    total_acc = tl.zeros([], dtype=tl.float32)

    # Outer loop over rows in this chunk
    for r in range(row_start, row_end):
        row_off = r * cols
        row_acc = tl.zeros([], dtype=tl.float32)

        # Inner loop over columns
        for col_start in range(0, cols, BLOCK_SIZE):
            offsets = col_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < cols

            q_vals = tl.load(
                q_ptr + row_off + offsets,
                mask=mask,
                other=0.0,
                eviction_policy='evict_first',
            )
            log_p_vals = tl.load(
                log_p_ptr + row_off + offsets,
                mask=mask,
                other=0.0,
                eviction_policy='evict_first',
            )

            term = tl.where(
                q_vals > 0.0,
                q_vals * (tl.log(q_vals) - log_p_vals),
                0.0,
            )
            row_acc += tl.sum(term)

        total_acc += row_acc

    tl.store(partial_sum_ptr + pid, total_acc)


@triton.jit
def reduce_partials_kernel(
    partial_sum_ptr,    # f32 [num_chunks]
    scalar_ptr,         # f32 [1]
    num_chunks: tl.constexpr,
    rows: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Single-block reduction that sums all partial totals and divides by
    the total number of rows to produce the batch-mean KL divergence.
    """
    total = tl.zeros([], dtype=tl.float32)
    for start in range(0, num_chunks, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_chunks
        vals = tl.load(
            partial_sum_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy='evict_first',
        )
        total += tl.sum(vals)

    tl.store(scalar_ptr, total / rows)


def triton_run(log_p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence with reduction='batchmean'.
    Inputs: log_p (f32[8192, 8192]), q (f32[8192, 8192])
    Returns: f32[] scalar
    """
    assert log_p.shape == q.shape, "Input shapes must match"
    assert log_p.dtype == q.dtype == torch.float32, "Inputs must be float32"
    rows, cols = log_p.shape

    # Hyperparameters: number of rows handled by one program.
    ROWS_PER_PROG = 8
    BLOCK_SIZE = 2048   # column tile size

    num_chunks = triton.cdiv(rows, ROWS_PER_PROG)

    partial_sums = torch.empty(num_chunks, dtype=torch.float32, device=log_p.device)
    scalar_out = torch.empty(1, dtype=torch.float32, device=log_p.device)

    # Kernel 1: chunked row processing
    kl_chunk_kernel[(num_chunks,)](
        log_p, q, partial_sums,
        rows=rows, cols=cols,
        ROWS_PER_PROG=ROWS_PER_PROG,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=16,
        num_stages=4,
    )

    # Kernel 2: reduction of partial sums
    REDUCE_BLOCK = min(1024, num_chunks)  # single iteration for small num_chunks
    reduce_partials_kernel[(1,)](
        partial_sums, scalar_out,
        num_chunks=num_chunks,
        rows=rows,
        BLOCK_SIZE=REDUCE_BLOCK,
        num_warps=4,
        num_stages=2,
    )

    return scalar_out.squeeze()