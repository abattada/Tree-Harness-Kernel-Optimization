import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice  # not used here but allowed

@triton.jit
def rms_norm_kernel(
    x_ptr,
    y_ptr,
    n_rows,
    n_cols,
    eps,
    row_stride,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    pid = tl.program_id(0)
    start_row = pid * ROWS_PER_PROGRAM
    end_row = tl.minimum(start_row + ROWS_PER_PROGRAM, n_rows)

    # Loop over rows assigned to this program
    for row_idx in range(start_row, end_row):
        # Compute base offset for this row
        offsets = row_idx * row_stride + tl.arange(0, BLOCK_SIZE)
        # Load the entire row (no mask because n_cols == BLOCK_SIZE)
        x = tl.load(x_ptr + offsets)
        # Compute sum of squares
        x_sq = x * x
        row_sum = tl.sum(x_sq, axis=0)  # scalar
        # Compute mean and scale
        mean = row_sum / n_cols
        scale = tl.rsqrt(mean + eps)
        # Apply scale and store
        y = x * scale
        tl.store(y_ptr + offsets, y)

def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous()
    assert x.dtype == torch.float32
    n_rows, n_cols = x.shape

    # Tuning knobs: can be adjusted later
    BLOCK_SIZE = 4096
    ROWS_PER_PROGRAM = 4
    assert n_cols == BLOCK_SIZE, "This kernel expects n_cols == BLOCK_SIZE for simplicity"

    y = torch.empty_like(x)

    grid = (triton.cdiv(n_rows, ROWS_PER_PROGRAM),)
    rms_norm_kernel[grid](
        x,
        y,
        n_rows,
        n_cols,
        1e-5,
        x.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        ROWS_PER_PROGRAM=ROWS_PER_PROGRAM,
    )
    return y