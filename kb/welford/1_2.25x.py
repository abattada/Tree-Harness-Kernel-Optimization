import torch
import triton
import triton.language as tl

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

    # offsets for the block (full row)
    offsets = tl.arange(0, BLOCK_SIZE)
    x_ptrs = x_ptr + row_start + offsets
    x = tl.load(x_ptrs)             # no mask needed because size divides block

    # single‑pass sum and sum of squares
    s = tl.sum(x, axis=0).to(tl.float32)
    sq = tl.sum(x * x, axis=0).to(tl.float32)

    mean = s / n_cols
    var = (sq / n_cols) - mean * mean   # population variance

    # store results
    out_stride = n_rows               # output is [2, n_rows]
    out_ptr_mean = out_ptr + 0 * out_stride + pid
    out_ptr_var  = out_ptr + 1 * out_stride + pid
    tl.store(out_ptr_mean, mean)
    tl.store(out_ptr_var, var)

def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32
    n_rows, n_cols = x.shape
    out = torch.empty((2, n_rows), dtype=torch.float32, device=x.device)

    BLOCK_SIZE = n_cols  # 4096, power of two, full row

    grid = (n_rows,)
    welford_kernel[grid](
        x,
        out,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out