import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Persistent kernel: each program processes a stride of rows, performs online
# softmax + NLL loss for each row, and accumulates a per-program partial sum.
# This eliminates the intermediate per-row loss array and the first reduction
# stage, while also reducing the total number of program launches.
# ---------------------------------------------------------------------------
@triton.jit
def cross_entropy_persistent_kernel(
    logits_ptr,              # float32 [R, N]
    targets_ptr,             # int64   [R]
    partials_ptr,            # float32 [grid_size] – partial sums per program
    R: tl.constexpr,         # total rows (8192)
    N: tl.constexpr,         # number of classes (32768)
    BLOCK_SIZE_N: tl.constexpr,  # block along class dimension (must divide N)
    grid_size: tl.constexpr,     # total number of programs
):
    pid = tl.program_id(0)
    partial_sum = tl.zeros([], dtype=tl.float32)

    # Grid‑stride loop over rows – each program handles a subset of rows
    for r in range(pid, R, grid_size):
        # Load target class for this row
        target = tl.load(targets_ptr + r)

        # Online softmax – compute max and log‑sum‑exp in one pass
        m_curr = tl.full([], float('-inf'), dtype=tl.float32)
        d_curr = tl.full([], 0.0, dtype=tl.float32)
        row_base = r * N

        for start in range(0, N, BLOCK_SIZE_N):
            offs = start + tl.arange(0, BLOCK_SIZE_N)
            mask = offs < N   # always true because BLOCK_SIZE_N divides N
            x = tl.load(
                logits_ptr + row_base + offs,
                mask=mask,
                other=float('-inf'),
                eviction_policy='evict_first',
            )

            m_loc = tl.max(x, axis=0)
            m_new = tl.maximum(m_curr, m_loc)
            exp_centered = tl.exp(x - m_new)
            sum_exp = tl.sum(exp_centered, axis=0)
            d_new = d_curr * tl.exp(m_curr - m_new) + sum_exp
            m_curr = m_new
            d_curr = d_new

        logsumexp = m_curr + tl.log(d_curr)
        target_logit = tl.load(logits_ptr + row_base + target)
        loss = logsumexp - target_logit
        partial_sum += loss

    tl.store(partials_ptr + pid, partial_sum)


# ---------------------------------------------------------------------------
# Final reduction kernel – sums the partial sums and divides by R.
# ---------------------------------------------------------------------------
@triton.jit
def final_mean_kernel(
    partials_ptr,          # float32 [num_partials]
    out_ptr,               # float32 [] – scalar output
    R: tl.constexpr,       # total rows (for division)
    num_partials: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total = tl.zeros([], dtype=tl.float32)

    for start in range(0, num_partials, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < num_partials
        vals = tl.load(partials_ptr + offs, mask=mask, other=0.0)
        total += tl.sum(vals, axis=0)

    mean = total / tl.full([], R, dtype=tl.float32)
    tl.store(out_ptr, mean)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def triton_run(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits : float32 [R, N]  (R = 8192, N = 32768)
    targets: int64   [R]
    returns: float32 scalar = mean cross-entropy loss
    """
    R, N = logits.shape
    assert targets.shape == (R,)
    assert logits.dtype == torch.float32 and targets.dtype == torch.int64
    assert N % 1024 == 0, "N must be divisible by BLOCK_SIZE_N (1024)"

    # Tuned constants
    BLOCK_SIZE_N = 1024
    grid_size = 1024          # number of persistent programs
    NUM_WARPS_MAIN = 8
    BLOCK_SIZE_MEAN = 256
    NUM_WARPS_MEAN = 4

    partials = torch.empty(grid_size, dtype=torch.float32, device=logits.device)

    # Step 1: persistent row processing + partial reduction
    cross_entropy_persistent_kernel[(grid_size,)](
        logits, targets, partials,
        R, N, BLOCK_SIZE_N, grid_size,
        num_warps=NUM_WARPS_MAIN,
    )

    # Step 2: final mean over partials
    out = torch.empty((), dtype=torch.float32, device=logits.device)
    final_mean_kernel[(1,)](
        partials, out,
        R, grid_size, BLOCK_SIZE_MEAN,
        num_warps=NUM_WARPS_MEAN,
    )

    return out