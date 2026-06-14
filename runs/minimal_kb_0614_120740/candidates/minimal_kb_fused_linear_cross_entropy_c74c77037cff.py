import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel: fused linear + cross-entropy loss for one row.
# Each program handles one row of x (4096 total). It iterates over blocks of
# weight rows (classes) and computes online softmax without materializing the
# full logit vector.
# ---------------------------------------------------------------------------
@triton.jit
def fused_linear_ce_row_kernel(
    x_ptr,          # f32 [rows, K]
    w_ptr,          # f32 [M, K]
    targets_ptr,    # i64 [rows]
    loss_row_ptr,   # f32 [rows]  (output per-row loss)
    M: tl.constexpr,           # number of classes (32768)
    K: tl.constexpr,           # feature dimension (2048)
    BLOCK_M: tl.constexpr,     # class block size (e.g., 64)
    BLOCK_K: tl.constexpr,     # feature block size (2048)
):
    pid = tl.program_id(0)
    row_start = pid * K  # offset into x for this row

    target = tl.load(targets_ptr + pid)

    # Online softmax state
    m_old = tl.full([], float('-inf'), dtype=tl.float32)
    d_old = tl.zeros([], dtype=tl.float32)
    target_logit = tl.zeros([], dtype=tl.float32)

    # Loop over class blocks
    for m_start in range(0, M, BLOCK_M):
        offsets_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offsets_m < M

        # Load tile of w: shape (BLOCK_M, BLOCK_K)
        # w is row-major: w[class][k]
        offs_k = tl.arange(0, BLOCK_K)
        w_ptrs = w_ptr + offsets_m[:, None] * K + offs_k[None, :]
        w_tile = tl.load(w_ptrs, mask=mask_m[:, None], other=0.0)

        # Load tile of x for this row: (BLOCK_K,)
        x_ptrs = x_ptr + row_start + offs_k
        x_tile = tl.load(x_ptrs)

        # Compute dot products for this block: sum over K
        # (BLOCK_M, BLOCK_K) * (BLOCK_K,) -> (BLOCK_M,)
        logits_block = tl.sum(w_tile * x_tile[None, :], axis=1)

        # Online softmax update
        m_loc = tl.max(logits_block, axis=0)
        m_new = tl.maximum(m_old, m_loc)
        # compute exp centered on m_new
        exp_centered = tl.exp(logits_block - m_new)
        sum_exp = tl.sum(exp_centered, axis=0)
        d_new = d_old * tl.exp(m_old - m_new) + sum_exp
        m_old = m_new
        d_old = d_new

        # Extract target logit if target lies in this block
        target_in_block = target - m_start
        cond = (target_in_block >= 0) & (target_in_block < BLOCK_M)
        # We need to load the appropriate element of logits_block.
        # Use tl.where to update only if condition holds.
        # Note: we index logits_block with a scalar tensor, which is valid.
        target_logit = tl.where(cond, logits_block[target_in_block], target_logit)

    # Compute final logsumexp and loss for this row
    logsumexp = m_old + tl.log(d_old)
    loss = logsumexp - target_logit
    tl.store(loss_row_ptr + pid, loss)


# ---------------------------------------------------------------------------
# Kernel 2: reduce per-row losses to a scalar mean.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_mean_kernel(
    inp_ptr,       # f32 [R]
    out_ptr,       # f32 [1]
    R: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    total = tl.zeros([], dtype=tl.float32)
    for start in range(0, R, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < R
        vals = tl.load(inp_ptr + offs, mask=mask, other=0.0)
        total += tl.sum(vals)
    tl.store(out_ptr, total / R)


def triton_run(x: torch.Tensor, w: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Fused linear + cross-entropy loss.
    x: (4096, 2048) float32
    w: (32768, 2048) float32
    targets: (4096,) int64
    returns scalar float32 (mean cross-entropy loss)
    """
    rows, K = x.shape
    M, _ = w.shape
    device = x.device
    # Allocate per-row loss array
    loss_row = torch.empty(rows, dtype=torch.float32, device=device)
    # Launch row kernel
    BLOCK_M = 64
    BLOCK_K = K  # 2048 (no tiling along K for simplicity)
    # We tile along M only; BLOCK_K = K ensures we load entire feature dimension each iteration.
    grid = (rows,)
    fused_linear_ce_row_kernel[grid](
        x, w, targets, loss_row,
        M=M, K=K,
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )
    # Single reduction kernel to compute mean
    out = torch.empty(1, dtype=torch.float32, device=device)
    BLOCK_REDUCE = 4096  # covers all rows in one iteration
    reduce_mean_kernel[(1,)](
        loss_row, out,
        R=rows, BLOCK_SIZE=BLOCK_REDUCE,
        num_warps=1,
    )
    return out