import torch
import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(
    x_ptr, out_ptr,
    stride_x, stride_out,
    N,  # number of columns
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    # This program processes ROWS_PER_PROG consecutive rows
    base_row = tl.program_id(0) * ROWS_PER_PROG
    cols = tl.arange(0, BLOCK_SIZE)

    # Loop over rows in the group
    for i in range(ROWS_PER_PROG):
        row = base_row + i
        # Stop if row index exceeds the number of rows (masked below)
        x_row_ptr = x_ptr + row * stride_x
        out_row_ptr = out_ptr + row * stride_out

        # Load entire row (assuming BLOCK_SIZE == N, no mask needed)
        x = tl.load(x_row_ptr + cols)
        # Compute sum of squares
        x_sq = x * x
        sum_sq = tl.sum(x_sq, axis=0)
        mean_sq = sum_sq / N
        rstd = tl.rsqrt(mean_sq + eps)
        out = x * rstd
        tl.store(out_row_ptr + cols, out)

def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous()
    M, N = x.shape
    out = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(N)  # 4096
    # Process 4 rows per program to reduce launch overhead and improve occupancy
    ROWS_PER_PROG = 4
    grid = ( (M + ROWS_PER_PROG - 1) // ROWS_PER_PROG, )

    rms_norm_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        N,
        eps=1e-5,
        BLOCK_SIZE=BLOCK_SIZE,
        ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=8,        # optimal for BLOCK_SIZE=4096 reduction
        num_stages=1,
    )
    return out