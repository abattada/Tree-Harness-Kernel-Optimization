import torch
import triton
import triton.language as tl

@triton.jit
def embedding_kernel(
    weight_ptr,
    idx_ptr,
    out_ptr,
    hidden: int,
    weight_stride: int,
    out_stride: int,
    n_rows: int,
    BLOCK_COLS: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * ROWS_PER_PROG
    end = tl.minimum(start + ROWS_PER_PROG, n_rows)

    cols = tl.arange(0, BLOCK_COLS)

    for i in range(start, end):
        idx_val = tl.load(idx_ptr + i)
        row_base = weight_ptr + idx_val * weight_stride
        # load the entire row (BLOCK_COLS == hidden, no mask needed)
        vals = tl.load(row_base + cols)
        out_base = out_ptr + i * out_stride
        tl.store(out_base + cols, vals)

def triton_run(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    assert weight.is_contiguous() and weight.dtype == torch.float32
    assert idx.is_contiguous() and idx.dtype == torch.int64
    n_rows = idx.shape[0]
    hidden = weight.shape[1]
    out = torch.empty((n_rows, hidden), dtype=torch.float32, device=weight.device)

    # Tune these constants: ROWS_PER_PROG can be increased for longer idx
    BLOCK_COLS = hidden  # load full row in one pass
    ROWS_PER_PROG = 16   # each program processes 16 indices
    grid = ((n_rows + ROWS_PER_PROG - 1) // ROWS_PER_PROG,)
    embedding_kernel[grid](
        weight.data_ptr(),
        idx.data_ptr(),
        out.data_ptr(),
        hidden,
        weight.stride(0),  # weight stride between rows (in elements)
        out.stride(0),
        n_rows,
        BLOCK_COLS,
        ROWS_PER_PROG,
        num_warps=4,
    )
    return out