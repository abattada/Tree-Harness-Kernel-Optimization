import torch
import triton
import triton.language as tl

@triton.jit
def welford_kernel(
    x_ptr,          # pointer to input [8192, 4096] float32
    out_ptr,        # pointer to output [2, 8192] float32
    N_COLS: tl.constexpr,  # 4096
    N_ROWS: tl.constexpr,  # 8192
):
    row = tl.program_id(0)

    # Each block processes exactly one row using one warp (32 threads)
    local_sum = 0.0
    local_sumsq = 0.0

    # Load the row in tiles of 32 elements; 4096 is a multiple of 32, so no mask needed
    base = 0
    while base < N_COLS:
        offs = base + tl.arange(0, 32)
        vals = tl.load(x_ptr + row * N_COLS + offs)  # coalesced load
        my_val = vals[tl.local_offset()]             # scalar for this thread
        local_sum += my_val
        local_sumsq += my_val * my_val
        base += 32

    # Warp-level reduction using shuffle down (no shared memory needed)
    sum_val = local_sum
    sumsq_val = local_sumsq
    for offset in [16, 8, 4, 2, 1]:
        sum_val   += tl.shfl_down(sum_val,   offset)
        sumsq_val += tl.shfl_down(sumsq_val, offset)

    # Lane 0 writes the final row statistics
    if tl.local_offset() == 0:
        mean = sum_val / N_COLS
        var = sumsq_val / N_COLS - mean * mean
        var = tl.maximum(var, 0.0)   # clamp tiny negative values from fp rounding
        out_mean_ptr = out_ptr + row
        out_var_ptr  = out_ptr + N_ROWS + row
        tl.store(out_mean_ptr, mean)
        tl.store(out_var_ptr, var)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """Compute per-row mean and population variance.
    Input:  x  float32 [8192, 4096]
    Output:     float32 [2, 8192]   (row 0 = mean, row 1 = variance)
    """
    assert x.shape == (8192, 4096)
    assert x.dtype == torch.float32
    out = torch.empty(2, 8192, dtype=x.dtype, device=x.device)

    # One block per row: block_size = 32 threads (1 warp)
    grid = (x.shape[0],)
    welford_kernel[grid](
        x, out,
        N_COLS=x.shape[1],
        N_ROWS=x.shape[0],
        num_warps=1,      # exactly one warp per block
    )
    return out