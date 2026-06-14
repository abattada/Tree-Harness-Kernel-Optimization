import torch
import triton
import triton.language as tl

# Tunable parameters
BLOCK_SIZE = 4096          # full row width, minus masking
NUM_WARPS = 4
NUM_STAGES = 4

@triton.jit
def welford_kernel_persistent(
    x_ptr,                     # input: [n_rows, n_cols]
    out_ptr,                   # output: [2, n_rows] (row0=mean, row1=var)
    n_rows: int,
    n_cols: int,
    rows_per_prog: int,        # number of rows each program should process
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    row_start = pid * rows_per_prog
    row_end = min(row_start + rows_per_prog, n_rows)

    # Each program loops over its assigned rows
    for row_idx in range(row_start, row_end):
        row_offset = row_idx * n_cols
        # Load full row – no mask because BLOCK_SIZE == n_cols exactly
        offsets = tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + row_offset + offsets, eviction_policy='evict_first')

        # Single-pass sum and sum of squares
        s = tl.sum(x, axis=0).to(tl.float32)
        sq = tl.sum(x * x, axis=0).to(tl.float32)

        mean = s / n_cols
        var = (sq / n_cols) - mean * mean   # population variance

        # Write per-row results
        out_ptr_mean = out_ptr + 0 * n_rows + row_idx
        out_ptr_var  = out_ptr + 1 * n_rows + row_idx
        tl.store(out_ptr_mean, mean)
        tl.store(out_ptr_var, var)

def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32
    n_rows, n_cols = x.shape
    out = torch.empty((2, n_rows), dtype=torch.float32, device=x.device)

    # Number of SMs on the target GPU (RTX 5090 has 170 SMs, but use detected count)
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    # Use a multiple of num_sms to keep occupancy high
    NUM_PROGRAMS = num_sms * 2   # typical choice for persistent kernel
    rows_per_prog = (n_rows + NUM_PROGRAMS - 1) // NUM_PROGRAMS

    grid = (NUM_PROGRAMS,)
    welford_kernel_persistent[grid](
        x, out,
        n_rows, n_cols,
        rows_per_prog,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return out