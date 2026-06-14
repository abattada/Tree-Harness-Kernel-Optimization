import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Fused per‑row online softmax + nll loss and partial sum reduction.
# Each program processes ROWS_PER_PROG rows and writes a partial sum.
# ---------------------------------------------------------------------------
@triton.jit
def cross_entropy_fused_kernel(
    logits_ptr,       # f32 [R, N]
    targets_ptr,      # i64 [R]
    out_partial_ptr,  # f32 [grid_size]  partial sums
    R: tl.constexpr,           # total rows
    N: tl.constexpr,           # number of classes
    BLOCK_SIZE_N: tl.constexpr,# tile along N
    ROWS_PER_PROG: tl.constexpr,# rows handled per program
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROG
    # Assume R is exactly divisible by ROWS_PER_PROG (8192 / 1024)
    tl.static_assert(R % ROWS_PER_PROG == 0, "R must be multiple of ROWS_PER_PROG")
    tl.static_assert(N % BLOCK_SIZE_N == 0, "N must be multiple of BLOCK_SIZE_N")

    acc = tl.zeros([], dtype=tl.float32)
    # Loop over rows in this chunk
    for row_offset in range(ROWS_PER_PROG):
        current_row = row_start + row_offset
        # Load target class
        target = tl.load(targets_ptr + current_row)

        # Online softmax state
        m = tl.full([], float('-inf'), dtype=tl.float32)
        d = tl.full([], 0.0, dtype=tl.float32)

        row_base = current_row * N
        # Process the row in BLOCK_SIZE_N tiles
        for start in range(0, N, BLOCK_SIZE_N):
            offs = start + tl.arange(0, BLOCK_SIZE_N)
            # No mask needed because N is exact multiple
            x = tl.load(logits_ptr + row_base + offs, eviction_policy='evict_first')

            m_loc = tl.max(x, axis=0)
            m_new = tl.maximum(m, m_loc)
            exp_centered = tl.exp(x - m_new)
            sum_exp = tl.sum(exp_centered, axis=0)
            d = d * tl.exp(m - m_new) + sum_exp
            m = m_new

        logsumexp = m + tl.log(d)
        # Load logit at target position
        target_logit = tl.load(logits_ptr + row_base + target, eviction_policy='evict_first')
        loss = logsumexp - target_logit
        acc += loss

    # Store partial sum for this group of rows
    tl.store(out_partial_ptr + pid, acc)


# ---------------------------------------------------------------------------
# Second reduction stage: sum all partials and compute mean.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_mean_stage2_kernel(
    inp_ptr,           # f32 [num_partials]
    out_scalar_ptr,    # f32 [1]
    R: tl.constexpr,               # total rows (for division)
    num_partials: tl.constexpr,    # number of partial sums
    BLOCK_SIZE: tl.constexpr,      # tile size (>= num_partials)
):
    total = tl.zeros([], dtype=tl.float32)
    for start in range(0, num_partials, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < num_partials
        vals = tl.load(inp_ptr + offs, mask=mask, other=0.0)
        total += tl.sum(vals)
    mean = total / R
    tl.store(out_scalar_ptr, mean)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def triton_run(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    R, N = logits.shape
    assert targets.shape == (R,)
    assert logits.is_cuda and targets.is_cuda

    # Fused kernel: one program per 1024 rows -> 8 programs
    ROWS_PER_PROG = 1024
    BLOCK_SIZE_N = 4096   # each thread: 4096/256 = 16 elements
    grid_fused = (R // ROWS_PER_PROG,)
    num_partials = grid_fused[0]
    partials = torch.empty(num_partials, dtype=torch.float32, device=logits.device)

    cross_entropy_fused_kernel[grid_fused](
        logits, targets, partials,
        R=R,
        N=N,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=8,
        num_stages=3,
    )

    # Second stage: combine partials and divide by R
    scalar_out = torch.empty((), dtype=torch.float32, device=logits.device)
    BLOCK_SIZE_RED = 1024  # big enough for 8 partials
    grid_red = (1,)
    reduce_mean_stage2_kernel[grid_red](
        partials, scalar_out,
        R=R,
        num_partials=num_partials,
        BLOCK_SIZE=BLOCK_SIZE_RED,
        num_warps=4,
        num_stages=2,
    )

    return scalar_out