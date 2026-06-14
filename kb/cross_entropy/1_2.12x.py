import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 1: per-row online softmax + nll loss, one program per row.
# Each row is processed in BLOCK_SIZE chunks.
# ---------------------------------------------------------------------------
@triton.jit
def cross_entropy_row_kernel(
    logits_ptr,   # f32 [R, N]
    targets_ptr,  # i64 [R]
    loss_row_ptr, # f32 [R]  (per‑row loss, to be reduced later)
    N: tl.constexpr,         # number of classes
    BLOCK_SIZE: tl.constexpr # block along the N dimension
):
    # Program id corresponds to the row index
    pid = tl.program_id(0)

    # Load the target class for this row
    target = tl.load(targets_ptr + pid)

    # ------------------------------------------------------------------
    # Online softmax: compute max and sum of exp in one pass.
    # Initialize with -inf for max and 0 for sum
    # ------------------------------------------------------------------
    m_old = tl.full([], float('-inf'), dtype=tl.float32)
    d_old = tl.full([], 0.0, dtype=tl.float32)

    # Loop over blocks along the N dimension
    for start in range(0, N, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        # Mask for the last block (though N is a multiple, keep for generality)
        mask = offs < N
        x = tl.load(logits_ptr + pid * N + offs, mask=mask, other=float('-inf'))

        # Local max and sum of exp(x - local_max)
        m_loc = tl.max(x, axis=0)
        # For the sum we need the elements after centering with m_loc,
        # but the online update uses m_new, not m_loc.
        # We compute m_new = max(m_old, m_loc) and then correct d.
        m_new = tl.maximum(m_old, m_loc)
        # Compute exp(x - m_new) and sum, for elements that are finite (masked ones are -inf -> exp(-inf)=0)
        exp_centered = tl.exp(x - m_new)
        sum_exp = tl.sum(exp_centered, axis=0)
        # Update d: the previous contributions need to be scaled by exp(m_old - m_new)
        d_new = d_old * tl.exp(m_old - m_new) + sum_exp
        m_old = m_new
        d_old = d_new

    # logsumexp = m_old + log(d_old)
    logsumexp = m_old + tl.log(d_old)

    # Load the logit at the target position
    target_logit = tl.load(logits_ptr + pid * N + target)

    # Loss for this row: logsumexp - logit[target]
    loss = logsumexp - target_logit

    # Store per‑row loss
    tl.store(loss_row_ptr + pid, loss)


# ---------------------------------------------------------------------------
# Kernel 2a: first reduction stage – each block sums a chunk of the
#            per‑row loss array and writes a partial sum.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_sum_stage1_kernel(
    inp_ptr,       # f32 [R]
    out_partial_ptr, # f32 [grid_size]
    R: tl.constexpr,          # total number of rows
    BLOCK_SIZE: tl.constexpr  # number of rows per program (must divide R)
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < R
    vals = tl.load(inp_ptr + offs, mask=mask, other=0.0)
    partial = tl.sum(vals, axis=0)
    tl.store(out_partial_ptr + pid, partial)


# ---------------------------------------------------------------------------
# Kernel 2b: final reduction – sum the partials and compute the mean.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_mean_stage2_kernel(
    partial_ptr,  # f32 [num_partials]
    out_ptr,      # f32 []
    num_partials: tl.constexpr,  # number of partial sums
    R: tl.constexpr              # total number of rows
):
    offs = tl.arange(0, num_partials)
    vals = tl.load(partial_ptr + offs, mask=offs < num_partials, other=0.0)
    total = tl.sum(vals, axis=0)
    mean = total / R
    tl.store(out_ptr, mean)


# ---------------------------------------------------------------------------
# Public entry point, as required by the contract.
# ---------------------------------------------------------------------------
def triton_run(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits: f32 [8192, 32768]
    targets: i64 [8192]
    returns: f32 []  (scalar, mean cross‑entropy loss)
    """
    R, N = logits.shape
    device = logits.device
    # Allocate per‑row loss array
    loss_row = torch.empty(R, dtype=torch.float32, device=device)

    # --- Kernel 1: per‑row processing ---
    BLOCK_SIZE = 1024  # 32768 / 1024 = 32 blocks, exact
    grid1 = (R,)       # one program per row
    cross_entropy_row_kernel[grid1](
        logits, targets, loss_row,
        N=N, BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4
    )

    # --- Two‑stage reduction: first stage ---
    BLOCK_RED1 = 1024
    num_partials = (R + BLOCK_RED1 - 1) // BLOCK_RED1   # = 8
    partials = torch.empty(num_partials, dtype=torch.float32, device=device)
    grid2a = (num_partials,)
    reduce_sum_stage1_kernel[grid2a](
        loss_row, partials,
        R=R, BLOCK_SIZE=BLOCK_RED1,
        num_warps=4
    )

    # --- Second stage: final reduction to scalar mean ---
    out = torch.empty((), dtype=torch.float32, device=device)  # scalar
    grid2b = (1,)
    reduce_mean_stage2_kernel[grid2b](
        partials, out,
        num_partials=num_partials, R=R,
        num_warps=1   # small block, 1 warp is enough
    )

    return out