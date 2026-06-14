import torch
import triton
import triton.language as tl


@triton.jit
def fused_linear_ce_kernel(
    x_ptr,
    w_ptr,
    target_ptr,
    loss_ptr,
    B,
    V,
    D,
    BLOCK_V: tl.constexpr,
    D_BLOCK: tl.constexpr,
):
    # Each program computes the cross-entropy loss for one row of the batch.
    row_idx = tl.program_id(0)
    if row_idx >= B:
        return

    target = tl.load(target_ptr + row_idx)

    # Running max and sum for the online softmax (cross-entropy) computation.
    # A large negative initial value is safe; it is quickly replaced by the true max.
    m = -1e10  # behaves like -inf
    s = 0.0
    t_logit = 0.0  # will eventually hold logit at target index

    # Iterate over vocabulary blocks.  V is divisible by BLOCK_V, so no boundary mask is needed.
    for v_start in range(0, V, BLOCK_V):
        v_offs = v_start + tl.arange(0, BLOCK_V)

        # Accumulate logits for this vocabulary block.
        logits = tl.zeros([BLOCK_V], dtype=tl.float32)

        # Tiled dot product over the hidden dimension.
        for d_start in range(0, D, D_BLOCK):
            d_offs = d_start + tl.arange(0, D_BLOCK)

            x_vals = tl.load(x_ptr + row_idx * D + d_offs)
            w_vals = tl.load(w_ptr + v_offs[:, None] * D + d_offs[None, :])

            # Compute partial dot products: (BLOCK_V, D_BLOCK) x (D_BLOCK,) -> (BLOCK_V,)
            logits += tl.sum(w_vals * x_vals[None, :], axis=1)

        # --- online softmax update ---
        block_max = tl.max(logits)
        new_m = tl.maximum(m, block_max)

        # Rescale running sum and add contributions from current block.
        s = s * tl.exp(m - new_m) + tl.sum(tl.exp(logits - new_m))
        m = new_m

        # If the target token lies in this block, record its logit.
        in_block = (v_start <= target) & (target < v_start + BLOCK_V)
        target_offset = target - v_start
        t_logit = tl.where(in_block, logits[target_offset], t_logit)

    # Final cross-entropy: log(sum) + global_max - target_logit
    loss = tl.log(s) + m - t_logit
    tl.store(loss_ptr + row_idx, loss)


def triton_run(x: torch.Tensor, w: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute the mean cross-entropy of x @ w.T without materialising the full
    logits tensor.  All inputs must reside on the same GPU device.

    Signature: triton_run(x: f32[4096, 2048], w: f32[32768, 2048], targets: i64[4096]) -> f32[]
    """
    B, D = x.shape
    V = w.shape[0]
    assert w.shape[1] == D, "w must have shape (V, D)"
    assert targets.shape == (B,), "targets must be 1D of length B"

    # Output buffer for per-sample losses; the caller will reduce them afterwards.
    loss_per_row = torch.empty(B, dtype=x.dtype, device=x.device)

    # Tile sizes that divide V=32768 and D=2048, avoiding boundary masks.
    BLOCK_V = 256
    D_BLOCK = 128
    grid = (B,)

    fused_linear_ce_kernel[grid](
        x, w, targets, loss_per_row,
        B, V, D,
        BLOCK_V=BLOCK_V,
        D_BLOCK=D_BLOCK,
        num_warps=8,
        num_stages=4,
    )

    # Mean reduction (trivial glue, permitted).
    return loss_per_row.mean()