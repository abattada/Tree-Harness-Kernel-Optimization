import torch
import triton
import triton.language as tl

# Fixed dimensions – the input is always (8192, 4096)
N_ROWS = 8192
N_COLS = 4096
BLOCK_SIZE = N_COLS  # one block covers an entire row; no masks needed
INV_N = 1.0 / N_COLS   # compile-time reciprocal

# Process two rows per program to reduce launch overhead and increase
# memory parallelism, while still keeping the kernel register-light.
ROWS_PER_PROG = 2


@triton.jit
def welford_kernel(
    x_ptr,                        # input [N_ROWS, N_COLS], float32
    out_ptr,                      # output [2, N_ROWS], float32
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
    INV_N: tl.constexpr,
):
    pid = tl.program_id(0)                # block index in the reduced grid
    row0 = pid * ROWS_PER_PROG            # first row handled by this block
    row1 = row0 + 1                       # second row (only valid when ROWS_PER_PROG==2)

    # Common offsets for a whole row (all elements)
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)

    # Load the two rows independently – 'evict_first' because they are used once.
    x0 = tl.load(x_ptr + row0 * n_cols + offsets,
                 eviction_policy='evict_first')
    x1 = tl.load(x_ptr + row1 * n_cols + offsets,
                 eviction_policy='evict_first')

    # Single-pass sum / sum-of-squares for each row (kept in fp32)
    s0  = tl.sum(x0, axis=0).to(tl.float32)
    sq0 = tl.sum(x0 * x0, axis=0).to(tl.float32)

    s1  = tl.sum(x1, axis=0).to(tl.float32)
    sq1 = tl.sum(x1 * x1, axis=0).to(tl.float32)

    # Population mean and variance using the precomputed inverse
    mean0 = s0 * INV_N
    var0  = (sq0 * INV_N) - mean0 * mean0

    mean1 = s1 * INV_N
    var1  = (sq1 * INV_N) - mean1 * mean1

    # Store results into the [2, n_rows] output tensor
    out_stride = n_rows
    tl.store(out_ptr + 0 * out_stride + row0, mean0)
    tl.store(out_ptr + 1 * out_stride + row0, var0)

    tl.store(out_ptr + 0 * out_stride + row1, mean1)
    tl.store(out_ptr + 1 * out_stride + row1, var1)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    Compute per-row mean and population variance of a float32 tensor.

    Equivalent to:
        mean = x.mean(dim=-1)
        var  = x.var(dim=-1, unbiased=False)
        return torch.stack([mean, var])

    Args:
        x: torch.Tensor of shape (8192, 4096), dtype float32, on CUDA.
    Returns:
        Out tensor of shape (2, 8192), dtype float32.
    """
    assert x.is_cuda and x.dtype == torch.float32
    assert x.shape == (N_ROWS, N_COLS), \
        f"Expected ({N_ROWS}, {N_COLS}) but got {x.shape}"
    assert N_ROWS % ROWS_PER_PROG == 0, \
        "Number of rows must be divisible by ROWS_PER_PROG"

    out = torch.empty((2, N_ROWS), dtype=torch.float32, device=x.device)

    grid = (N_ROWS // ROWS_PER_PROG,)
    welford_kernel[grid](
        x, out,
        n_rows=N_ROWS,
        n_cols=N_COLS,
        BLOCK_SIZE=BLOCK_SIZE,
        ROWS_PER_PROG=ROWS_PER_PROG,
        INV_N=INV_N,
        num_warps=8,         # 8 warps gives good latency hiding for this pattern
        num_stages=2,        # low stages keep register pressure minimal
    )
    return out