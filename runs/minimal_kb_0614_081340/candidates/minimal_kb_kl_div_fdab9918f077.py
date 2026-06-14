import torch
import triton
import triton.language as tl

@triton.jit
def fused_kl_kernel(
    log_p_ptr,  # [rows, cols]
    q_ptr,      # [rows, cols]
    scalar_ptr, # [1] (output)
    rows: tl.constexpr,
    cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROG
    total = tl.zeros([], dtype=tl.float32)

    # Loop over rows assigned to this program
    for r in range(ROWS_PER_PROG):
        row = row_start + r
        if row < rows:
            row_acc = tl.zeros([], dtype=tl.float32)
            # Process columns in tiles
            for col_start in range(0, cols, BLOCK_SIZE):
                offsets = col_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < cols
                q_vals = tl.load(q_ptr + row * cols + offsets, mask, other=0.0)
                log_p_vals = tl.load(log_p_ptr + row * cols + offsets, mask, other=0.0)
                term = tl.where(q_vals > 0, q_vals * (tl.log(q_vals) - log_p_vals), 0.0)
                row_acc += tl.sum(term)
            # row_acc is now a scalar (broadcasted) for this row
            total += row_acc

    # Atomically add this block's contribution to the global total
    tl.atomic_add(scalar_ptr, total)


def triton_run(log_p, q) -> torch.Tensor:
    assert log_p.shape == q.shape
    rows, cols = log_p.shape
    device = log_p.device

    # Output scalar, initialized to zero (for atomic accumulation)
    scalar_out = torch.zeros(1, dtype=torch.float32, device=device)

    # Tuned parameters
    BLOCK_SIZE = 1024
    ROWS_PER_PROG = 32            # Process 32 rows per block → 8192/32 = 256 blocks
    num_warps = 8

    grid = (triton.cdiv(rows, ROWS_PER_PROG),)
    fused_kl_kernel[grid](
        log_p, q, scalar_out,
        rows, cols, BLOCK_SIZE, ROWS_PER_PROG,
        num_warps=num_warps,
    )

    # Finalize: divide by rows for batchmean reduction
    result = scalar_out.squeeze() / rows
    return result

CONF: {"confidence": 85, "notes": "Fused row-computation and reduction into one kernel using atomic accumulated total, reducing kernel launches and intermediate memory traffic while retaining high memory bandwidth utilization."}