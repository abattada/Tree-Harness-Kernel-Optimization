import torch
import triton
import triton.language as tl


@triton.jit
def _flce_row_kernel(
    x_ptr,          # [B, D] f32, contiguous
    w_ptr,          # [V, D] f32, contiguous
    targets_ptr,    # [B]    i64
    output_sum_ptr, # scalar f32
    B,
    V,
    D,
    BLOCK_V: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    # each program handles one row (one sample)
    x_row_ptr = x_ptr + row * D

    # online logsumexp state
    row_max = float('-inf')
    row_sumexp = 0.0

    # ----- pass over the vocabulary dimension V -----
    for v_start in range(0, V, BLOCK_V):
        v_offs = v_start + tl.arange(0, BLOCK_V)
        v_mask = v_offs < V

        # accumulate logits for this tile of V
        logits = tl.zeros((BLOCK_V,), dtype=tl.float32)
        for d_start in range(0, D, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offs < D

            x_chunk = tl.load(x_row_ptr + d_offs, mask=d_mask, other=0.0)
            w_tile_ptr = w_ptr + v_offs[:, None] * D + d_offs[None, :]
            w_tile = tl.load(w_tile_ptr, mask=v_mask[:, None] & d_mask[None, :], other=0.0)
            logits += tl.sum(x_chunk[None, :] * w_tile, axis=1)

        # tile statistics
        masked_logits = tl.where(v_mask, logits, float('-inf'))
        tile_max = tl.max(masked_logits, axis=0)
        tile_sumexp = tl.sum(tl.where(v_mask, tl.exp(logits - tile_max), 0.0), axis=0)

        if v_start == 0:
            row_max = tile_max
            row_sumexp = tile_sumexp
        else:
            new_max = tl.maximum(row_max, tile_max)
            row_sumexp = row_sumexp * tl.exp(row_max - new_max) + tile_sumexp * tl.exp(tile_max - new_max)
            row_max = new_max

    # ----- compute logit for the target class -----
    t = tl.load(targets_ptr + row).to(tl.int32)
    logit_target = tl.zeros((), dtype=tl.float32)
    for d_start in range(0, D, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < D
        x_chunk = tl.load(x_row_ptr + d_offs, mask=d_mask, other=0.0)
        w_target_chunk = tl.load(w_ptr + t * D + d_offs, mask=d_mask, other=0.0)
        logit_target += tl.sum(x_chunk * w_target_chunk)

    logsumexp = row_max + tl.log(row_sumexp)
    loss = logsumexp - logit_target   # = logsumexp - logit_target
    tl.atomic_add(output_sum_ptr, loss)


def triton_run(x: torch.Tensor, w: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    x:      f32[4096, 2048]
    w:      f32[32768, 2048]
    targets: i64[4096]
    returns: f32[]  mean cross-entropy loss
    """
    # ensure contiguous layout (cheap glue, allowed)
    x = x.contiguous()
    w = w.contiguous()
    targets = targets.contiguous()

    B, D = x.shape
    V, D_w = w.shape
    assert D == D_w, "Dimension mismatch"
    assert targets.shape == (B,), f"targets shape {targets.shape}"

    output_sum = torch.zeros([], device=x.device, dtype=torch.float32)

    BLOCK_V = 128
    BLOCK_D = 64
    grid = (B,)

    _flce_row_kernel[grid](
        x, w, targets, output_sum,
        B, V, D,
        BLOCK_V=BLOCK_V,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=2,
    )

    return output_sum / B