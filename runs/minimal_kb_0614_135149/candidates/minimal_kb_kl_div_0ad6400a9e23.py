import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Fused kernel: each program (block) processes one full row (8192 elements),
# computes the row-wise KL divergence sum, and atomically adds it to a
# global accumulator.  This eliminates the intermediate row_sum buffer and the
# second reduction kernel, reducing global memory traffic and launch overhead.
# ---------------------------------------------------------------------------
@triton.jit
def fused_kl_kernel(
    log_p_ptr,          # f32 [rows, cols]
    q_ptr,              # f32 [rows, cols]
    global_total_ptr,   # f32 [1]  – accumulator for the total sum
    rows: tl.constexpr,
    cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * cols

    # All threads load contiguous segments of the row
    offsets = tl.arange(0, BLOCK_SIZE)

    # Coalesced loads with streaming cache hints (evict_first)
    q_vals = tl.load(
        q_ptr + row_start + offsets,
        eviction_policy='evict_first'
    )
    log_p_vals = tl.load(
        log_p_ptr + row_start + offsets,
        eviction_policy='evict_first'
    )

    # term = q * (log(q) - log_p), handling q == 0 safely
    term = tl.where(
        q_vals > 0.0,
        q_vals * (tl.log(q_vals) - log_p_vals),
        0.0
    )

    # Block-wide row sum (tree reduction in shared memory)
    row_sum = tl.sum(term)

    # Only thread 0 of each block atomically accumulates the row sum into the
    # global total.  This serialises 8192 atomic adds, which is well within
    # Turing/Blackwell throughput.
    if tl.linear_id(1) == 0:
        tl.atomic_add(global_total_ptr, row_sum)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def triton_run(log_p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Kullback–Leibler divergence with reduction='batchmean':
        sum(q * (log q - log_p)) / batch_size.
    Inputs: both [8192, 8192], float32.
    Returns: scalar float32 tensor.
    """
    rows, cols = log_p.shape
    assert log_p.shape == q.shape, "Input shapes must match"
    assert log_p.dtype == torch.float32, "Inputs must be float32"

    # Global accumulator initialised to zero
    global_total = torch.zeros(1, dtype=torch.float32, device=log_p.device)

    # Compile-time constants
    ROWS: tl.constexpr = rows
    COLS: tl.constexpr = cols
    BLOCK_SIZE: tl.constexpr = 8192

    # Launch one program (block) per row; each block handles the entire row
    fused_kl_kernel[(ROWS,)](
        log_p, q, global_total,
        rows=ROWS, cols=COLS, BLOCK_SIZE=BLOCK_SIZE,
        num_warps=16,
    )

    # Batchmean reduction: divide by number of rows (lazily on GPU)
    batchmean = global_total[0] / rows
    return batchmean