import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def geglu_kernel(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols_out,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row indices
    rows_offs = tl.arange(0, BLOCK_M) + pid_m * BLOCK_M
    rows_mask = rows_offs < n_rows

    # Column indices for output (a part)
    col_offs = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    col_mask = col_offs < n_cols_out

    # Input pointer offsets: a is at same column index; b is at column + n_cols_out
    # Base pointers for this row block (row-major)
    input_base = input_ptr + rows_offs[:, None] * 2 * n_cols_out  # input has 2*n_cols_out columns
    output_base = output_ptr + rows_offs[:, None] * n_cols_out

    # Load a and b tensors
    a = tl.load(input_base + col_offs[None, :], mask=rows_mask[:, None] & col_mask[None, :], other=0.0)
    b = tl.load(input_base + col_offs[None, :] + n_cols_out, mask=rows_mask[:, None] & col_mask[None, :], other=0.0)

    # Compute GELU with tanh approximation on a
    # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/π)
    x = a
    x3 = x * x * x
    inner = sqrt_2_over_pi * (x + 0.044715 * x3)
    tanh_val = libdevice.tanh(inner)
    gelu_a = 0.5 * x * (1.0 + tanh_val)

    # Multiply by b
    out = gelu_a * b

    # Store output
    tl.store(output_base + col_offs[None, :], out, mask=rows_mask[:, None] & col_mask[None, :])

def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32
    assert x.shape == (8192, 8192)
    n_rows, n_cols = x.shape
    n_cols_out = n_cols // 2  # 4096

    output = torch.empty(n_rows, n_cols_out, device=x.device, dtype=torch.float32)

    # Tuneable block sizes – favour occupancy: smaller blocks → more programs
    BLOCK_M = 128
    BLOCK_N = 256

    grid = (triton.cdiv(n_rows, BLOCK_M), triton.cdiv(n_cols_out, BLOCK_N))
    geglu_kernel[grid](
        x, output,
        n_rows, n_cols_out,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return output