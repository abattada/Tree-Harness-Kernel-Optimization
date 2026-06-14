import torch
import triton
import triton.language as tl

# Kernel: one row per program, one-pass sum & sum-of-squares
@triton.jit
def welford_kernel(
    x_ptr,         # input [8192, 4096]
    out_ptr,       # output [2, 8192]   (row0=mean, row1=variance)
    stride_x_row,  # 4096
    stride_out_0,  # 8192
    stride_out_1,  # 1
    BLOCK_SIZE: tl.constexpr,
    N: tl.constexpr,
):
    pid = tl.program_id(0)                     # row index
    row_start = pid * stride_x_row

    # scalar accumulators
    sum_val = tl.zeros([1], dtype=tl.float32)
    sum_sq_val = tl.zeros([1], dtype=tl.float32)

    # grid-stride loop over the row
    offsets = tl.arange(0, BLOCK_SIZE)
    for start in range(0, N, BLOCK_SIZE):
        x_ptrs = x_ptr + row_start + start + offsets
        x = tl.load(x_ptrs)                    # no mask needed; N is multiple of BLOCK_SIZE
        sum_val += tl.sum(x)
        sum_sq_val += tl.sum(x * x)

    mean = sum_val / N
    var = (sum_sq_val / N) - (mean * mean)

    # write results
    out_mean_ptr = out_ptr + pid * stride_out_1 + 0 * stride_out_0
    out_var_ptr  = out_ptr + pid * stride_out_1 + 1 * stride_out_0
    tl.store(out_mean_ptr, mean)
    tl.store(out_var_ptr, var)

def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32
    n_rows, n_cols = x.shape
    out = torch.empty((2, n_rows), dtype=torch.float32, device='cuda')

    BLOCK_SIZE = 1024  # must divide n_cols
    assert n_cols % BLOCK_SIZE == 0, "n_cols must be multiple of BLOCK_SIZE"

    # launch one program per row
    grid = (n_rows,)
    welford_kernel[grid](
        x,
        out,
        x.stride(0),      # row stride
        out.stride(0),
        out.stride(1),
        BLOCK_SIZE,
        n_cols,
    )
    return out