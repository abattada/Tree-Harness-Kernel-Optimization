import torch
import triton
import triton.language as tl

@triton.jit
def fused_linear_ce_kernel(
    x_ptr, w_ptr, targets_ptr, output_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    rows_start = pid * BLOCK_M

    # Load target indices for this row block (all rows are valid, M divides BLOCK_M)
    target_offs = rows_start + tl.arange(0, BLOCK_M)
    targets = tl.load(targets_ptr + target_offs)

    # Online softmax accumulators and target logit accumulator
    m = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    tgt_logit = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Loop over column tiles
    for n_start in range(0, N, BLOCK_N):
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # K loop – full matrix multiply tile
        for k in range(0, K, BLOCK_K):
            x_offs = (rows_start + tl.arange(0, BLOCK_M)[:, None]) * K + \
                     (k + tl.arange(0, BLOCK_K)[None, :])
            x_tile = tl.load(x_ptr + x_offs)

            w_offs = (n_start + tl.arange(0, BLOCK_N)[:, None]) * K + \
                     (k + tl.arange(0, BLOCK_K)[None, :])
            w_tile = tl.load(w_ptr + w_offs)

            acc += tl.dot(x_tile, w_tile)

        # Update running max and sum of exponentials
        tile_max = tl.max(acc, axis=1)                     # (BLOCK_M,)
        new_m = tl.maximum(m, tile_max)
        exp_vals = tl.exp(acc - new_m[:, None])
        tile_sum = tl.sum(exp_vals, axis=1)                # (BLOCK_M,)
        l = l * tl.exp(m - new_m) + tile_sum
        m = new_m

        # If the target column falls into this tile, record the full logit
        tgt_local = targets - n_start
        in_tile = (tgt_local >= 0) & (tgt_local < BLOCK_N)
        # Safely index into acc for rows that have their target in this tile
        safe_local = tl.where(in_tile, tgt_local, 0)
        val = tl.where(in_tile, acc[tl.arange(0, BLOCK_M), safe_local], tgt_logit)
        tgt_logit = tl.where(in_tile, val, tgt_logit)

    # Final per-row NLL and atomic sum
    logsumexp = m + tl.log(l)
    nll = logsumexp - tgt_logit
    nll_sum = tl.sum(nll)
    tl.atomic_add(output_ptr, nll_sum)


def triton_run(x: torch.Tensor, w: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    M, K = x.shape      # 4096 × 2048
    N, K2 = w.shape     # 32768 × 2048
    assert K == K2, "Inner dimensions must match"

    output = torch.zeros(1, dtype=torch.float32, device=x.device)

    # Use block sizes that are divisors of the known problem dimensions:
    # 4096 ÷ 64 = 64, 32768 ÷ 128 = 256, 2048 ÷ 128 = 16  (all exact)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 128, 128
    grid = (M // BLOCK_M,)

    fused_linear_ce_kernel[grid](
        x, w, targets, output,
        M, N, K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    # Mean CE over all rows
    return output[0] / M