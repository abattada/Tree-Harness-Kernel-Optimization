import torch
import triton
import triton.language as tl

# Fixed problem sizes — fully specialized at compile time.
N_ROWS: tl.constexpr = 8192
N_COLS: tl.constexpr = 4096
INV_N: tl.constexpr = 1.0 / float(N_COLS)  # precomputed reciprocal for faster division

@triton.jit
def welford_kernel(
    x_ptr,                # input  shape [N_ROWS, N_COLS], float32, contiguous
    out_ptr,              # output shape [2, N_ROWS], float32
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INV_N: tl.constexpr,
):
    pid = tl.program_id(0)               # row index
    row_start = pid * n_cols

    # Vectorized load of the whole row. max_contiguous hints the compiler to
    # generate wide memory transactions; evict_first avoids polluting caches
    # for data that is read exactly once.
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
    x_ptrs = x_ptr + row_start + offsets
    x = tl.load(x_ptrs, eviction_policy='evict_first')

    # Single-pass sum and sum-of-squares — all in float32 for numerical accuracy.
    s = tl.sum(x, axis=0).to(tl.float32)
    sq = tl.sum(x * x, axis=0).to(tl.float32)

    # Population mean and variance.
    mean = s * INV_N
    var = (sq * INV_N) - mean * mean

    # Store results. Layout: out[0, :] = mean, out[1, :] = var.
    out_stride = n_rows
    tl.store(out_ptr + 0 * out_stride + pid, mean)
    tl.store(out_ptr + 1 * out_stride + pid, var)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    Compute per-row mean and population variance for a float32 tensor
    of shape (8192, 4096). Returns a tensor of shape (2, 8192) where
    row 0 is the mean and row 1 is the population variance.
    """
    # Quick sanity checks — the harness guarantees these.
    assert x.is_cuda and x.dtype == torch.float32
    assert x.shape == (N_ROWS, N_COLS), f"Expected ({N_ROWS},{N_COLS}), got {x.shape}"

    out = torch.empty((2, N_ROWS), dtype=torch.float32, device=x.device)

    # One thread-block per row (8192 blocks), each processing an entire row (4096 elements).
    grid = (N_ROWS,)
    welford_kernel[grid](
        x,
        out,
        n_rows=N_ROWS,
        n_cols=N_COLS,
        BLOCK_SIZE=N_COLS,
        INV_N=INV_N,
        num_warps=4,     # 128 threads per block, good balance for memory-bound kernel
        num_stages=2,    # minimal stages keep register pressure low
    )
    return out