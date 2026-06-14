import torch
import triton
import triton.language as tl
import math

# ---------------------------------------------------------------------------
# Row-wise cross-entropy kernel: online softmax, single pass per row.
# Each program handles one row of logits [num_classes].
# Writes per-row loss (scalar) into row_losses array.
# ---------------------------------------------------------------------------
@triton.jit
def cross_entropy_row_kernel(
    logits_ptr,        # [num_rows, num_classes] row-major
    targets_ptr,       # [num_rows] int64
    row_losses_ptr,    # [num_rows] output (float32)
    num_rows: tl.constexpr,
    num_classes: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    # 1. Load target index and the corresponding logit
    target = tl.load(targets_ptr + pid, mask=pid < num_rows, other=0)  # int64
    # logits[pid, target]
    target_offset = pid * num_classes + target
    target_logit = tl.load(logits_ptr + target_offset)

    # 2. Online softmax over the row
    prev_max = -float('inf')
    d = 0.0

    # Loop over tiles of the row
    for start in range(0, num_classes, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_classes
        x_ptr = logits_ptr + pid * num_classes + offsets

        # ----- Load for max -----
        x_max = tl.load(x_ptr, mask=mask, other=-float('inf'))
        tile_max = tl.max(x_max, axis=0)

        # ----- Compute new max -----
        new_max = tl.maximum(prev_max, tile_max)

        # ----- Scale previous d -----
        scale_prev = tl.exp(prev_max - new_max)

        # ----- Load again for exp/sum (with zero filling) -----
        x_exp = tl.load(x_ptr, mask=mask, other=0.0)
        # exp(x - new_max)
        x_shifted = tl.exp(x_exp - new_max)
        tile_sum = tl.sum(x_shifted, axis=0, mask=mask)

        # Update running quantities
        d = d * scale_prev + tile_sum
        prev_max = new_max

    # 3. Compute logsumexp and loss
    logsumexp = prev_max + tl.log(d)
    loss = logsumexp - target_logit

    # 4. Write per-row loss
    tl.store(row_losses_ptr + pid, loss, mask=pid < num_rows)


# ---------------------------------------------------------------------------
# Reduction kernel: sum over all per-row losses, using atomic add.
# Block size must be a power of two.
# ---------------------------------------------------------------------------
@triton.jit
def sum_reduce_kernel(
    vec_ptr,       # [N] float32
    out_ptr,       # [1] float32
    N: tl.constexpr,
    BLOCK_REDUCE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_REDUCE
    offsets = block_start + tl.arange(0, BLOCK_REDUCE)
    mask = offsets < N
    x = tl.load(vec_ptr + offsets, mask=mask, other=0.0)
    # Warp-level reduction using tl.sum (assumes one warp)
    # For larger blocks, we'd do a tree; here BLOCK_REDUCE<=1024, single warp ok.
    block_sum = tl.sum(x, axis=0, mask=mask)
    # Atomic add to global output
    # Launch one block per chunk; guarantee only one block per chunk.
    prev = tl.atomic_add(out_ptr, block_sum, sem='relaxed')


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def triton_run(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    assert logits.is_contiguous()
    assert targets.is_contiguous()
    num_rows, num_classes = logits.shape
    assert targets.shape == (num_rows,)
    assert logits.dtype == torch.float32
    assert targets.dtype == torch.int64

    # ---------- 1. Row-wise cross-entropy ----------
    row_losses = torch.empty(num_rows, dtype=torch.float32, device=logits.device)
    BLOCK_SIZE = 1024
    grid_row = (num_rows,)  # one program per row
    cross_entropy_row_kernel[grid_row](
        logits, targets, row_losses,
        num_rows, num_classes,
        BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
    )

    # ---------- 2. Reduce to scalar (mean) ----------
    output = torch.empty(1, dtype=torch.float32, device=logits.device)
    output.zero_()  # initialise for atomic add
    BLOCK_REDUCE = 1024
    grid_reduce = (triton.cdiv(num_rows, BLOCK_REDUCE),)
    sum_reduce_kernel[grid_reduce](
        row_losses, output,
        num_rows,
        BLOCK_REDUCE,
        num_warps=4,
        num_stages=2,
    )
    # Divide by number of rows
    output[0] /= num_rows
    return output