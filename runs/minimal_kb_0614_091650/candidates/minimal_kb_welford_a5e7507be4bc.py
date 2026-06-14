import torch
import triton
import triton.language as tl

# Tunable parameters — optimal for 4096 columns, full row per program
BLOCK_SIZE = 4096
NUM_WARPS = 4
NUM_STAGES = 4

@triton.jit
def welford_kernel(
    x_ptr,                          # input: (n_rows, n_cols)
    out_ptr,                        # output: (2, n_rows)
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)          # row index
    row_start = pid * n_cols

    # Offsets for the entire row — no mask needed since BLOCK_SIZE divides n_cols
    offsets = tl.arange(0, BLOCK_SIZE)

    # Load with alignment and contiguity hints for the compiler
    x_ptrs = x_ptr + row_start + offsets
    tl.multiple_of(x_ptrs, BLOCK_SIZE * 4)   # 4096 floats = 16384 bytes, 128-byte aligned
    tl.max_contiguous(x_ptrs, BLOCK_SIZE * 4)
    x = tl.load(x_ptrs)

    # Single‑pass sum and sum of squares
    s = tl.sum(x, axis=0).to(tl.float32)
    sq = tl.sum(x * x, axis=0).to(tl.float32)

    mean = s / n_cols
    var = (sq / n_cols) - mean * mean   # population variance

    # Stride for output: output shape [2, n_rows]
    out_stride = n_rows
    out_ptr_mean = out_ptr + 0 * out_stride + pid
    out_ptr_var  = out_ptr + 1 * out_stride + pid
    tl.store(out_ptr_mean, mean)
    tl.store(out_ptr_var, var)

def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32
    n_rows, n_cols = x.shape
    out = torch.empty((2, n_rows), dtype=torch.float32, device=x.device)

    assert n_cols == BLOCK_SIZE  # Must be exact multiple; here it's 4096

    grid = (n_rows,)
    welford_kernel[grid](
        x,
        out,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return out