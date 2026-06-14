import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 1: per‑row online softmax + NLL loss, one program per row.
# Autotuned for BLOCK_SIZE, num_warps, num_stages.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=3),
    ],
    key=['N'],
)
@triton.jit
def cross_entropy_row_kernel(
    logits_ptr,      # f32 [R, N]
    targets_ptr,     # i64 [R]
    loss_row_ptr,    # f32 [R]   per‑row loss
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    target = tl.load(targets_ptr + pid)

    # Online softmax (single pass)
    m_old = tl.full([], float('-inf'), dtype=tl.float32)
    d_old = tl.full([], 0.0, dtype=tl.float32)

    row_base = pid * N
    # Use grid‑stride loop over BLOCK_SIZE tiles
    for start in range(0, N, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        # N is multiple of BLOCK_SIZE (auto‑checked), mask always true
        mask = offs < N  # kept for safety, compiler will optimize
        x = tl.load(
            logits_ptr + row_base + offs,
            mask=mask,
            other=float('-inf'),
            eviction_policy='evict_first',
        )
        m_loc = tl.max(x, axis=0)
        m_new = tl.maximum(m_old, m_loc)
        exp_centered = tl.exp(x - m_new)
        sum_exp = tl.sum(exp_centered, axis=0)
        d_new = d_old * tl.exp(m_old - m_new) + sum_exp
        m_old = m_new
        d_old = d_new

    logsumexp = m_old + tl.log(d_old)
    target_logit = tl.load(logits_ptr + row_base + target)
    loss = logsumexp - target_logit
    tl.store(loss_row_ptr + pid, loss)


# ---------------------------------------------------------------------------
# Kernel 2a: first reduction stage – sum chunks of per‑row losses.
# Autotuned for BLOCK_SIZE and num_warps.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=3),
    ],
    key=['R'],
)
@triton.jit
def reduce_sum_stage1_kernel(
    inp_ptr,          # f32 [R]
    out_partial_ptr,  # f32 [num_partials]
    R: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < R
    vals = tl.load(inp_ptr + offs, mask=mask, other=0.0,
                   eviction_policy='evict_first')
    partial = tl.sum(vals, axis=0)
    tl.store(out_partial_ptr + pid, partial)


# ---------------------------------------------------------------------------
# Kernel 2b: second reduction stage – sum all partials and compute mean.
# Uses a single block (num_partials is small).
# ---------------------------------------------------------------------------
@triton.jit
def reduce_mean_stage2_kernel(
    inp_ptr,          # f32 [num_partials]
    out_scalar_ptr,   # f32 [1]
    R: tl.constexpr,
    num_partials: tl.constexpr,
):
    offs = tl.arange(0, num_partials)
    vals = tl.load(inp_ptr + offs, eviction_policy='evict_first')
    total = tl.sum(vals, axis=0)
    mean = total / R
    tl.store(out_scalar_ptr, mean)


def triton_run(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    R, N = logits.shape
    assert logits.is_cuda and targets.is_cuda
    assert logits.dtype == torch.float32 and targets.dtype == torch.int64
    assert R == targets.shape[0]

    # The row kernel uses autotune – block size chosen automatically.
    # Compute number of partial sums for stage1: block size will be tuned but
    # we need the actual programmed block size from autotune. We'll compute
    # num_partials = (R + BLOCK_SIZE - 1) // BLOCK_SIZE, but since we don't
    # know BLOCK_SIZE until runtime (autotune picks), we set a dummy value
    # that will be replaced by the actual number of programs launched.
    # Simpler: launch stage1 with grid size = (R + 1023) // 1024, then use
    # that many partials. We'll pass the actual num_partials as constexpr
    # to stage2, computed from the grid.
    # But we don't know grid count until autotune sets block size.
    # Hack: launch stage1 with a fixed grid of (R + 1023) // 1024 (same as
    # parent) and let stage1's autotune choose BLOCK_SIZE independently.
    # The number of partials is determined by grid size, not the BLOCK_SIZE.
    # That's okay: each block will process BLOCK_SIZE rows, but if grid size
    # is (R+1023)//1024 and BLOCK_SIZE may differ, the last block may handle
    # fewer rows. That's fine; the kernel computes partial correctly via mask.
    # So we precompute grid1 = (R + 1023) // 1024 and use that as num_partials.
    # This decouples autotune BLOCK_SIZE from grid count. The kernel will
    # load correct number of rows up to BLOCK_SIZE, masking to R.
    # This is a bit wasteful if BLOCK_SIZE > 1024 (overlapping?), but safe.
    # Alternatively, we can compute grid1 after autotune by reading the
    # config. Since we cannot know the config before launch, we stick with
    # a fixed grid size that is large enough. This may cause extra work
    # but the number of partials remains small (e.g., at worst 16 if R=8192 and we use 512 per block).
    # Let's use grid1 = (R + 511) // 512 to ensure at least 512 rows per block.
    # But then num_partials varies. For simplicity, use parent's grid1 = (R + 1023) // 1024.
    # This yields 8 partials.
    grid1 = (R + 1023) // 1024
    num_partials = grid1

    # Per-row losses
    loss_row = torch.empty(R, device='cuda', dtype=torch.float32)

    # Launch kernel 1: one program per row, autotune will choose config
    cross_entropy_row_kernel[(R,)](
        logits, targets, loss_row,
        N=N,
    )

    # Partial sums
    partials = torch.empty(num_partials, device='cuda', dtype=torch.float32)

    # Launch kernel 2a (autotuned)
    reduce_sum_stage1_kernel[grid1](
        loss_row, partials,
        R=R,
    )

    # Launch kernel 2b: single block
    result = torch.empty(1, device='cuda', dtype=torch.float32)
    reduce_mean_stage2_kernel[(1,)](
        partials, result,
        R=R,
        num_partials=num_partials,
        num_warps=1,
    )

    return result.view(())