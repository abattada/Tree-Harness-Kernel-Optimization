import torch
import triton
import triton.language as tl


@triton.jit
def swiglu_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_x0,
    stride_out0,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    V: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Number of columns handled per thread-block  
    N_THREADS_COL = BLOCK_N // V
    # Total threads = BLOCK_M * N_THREADS_COL, each thread has a unique tid
    tid = tl.arange(0, BLOCK_M * N_THREADS_COL)

    row_idx = tid // N_THREADS_COL
    col_base = (tid % N_THREADS_COL) * V

    g_row = pid_m * BLOCK_M + row_idx
    g_col_start = pid_n * BLOCK_N + col_base

    offs = tl.arange(0, V)

    # Compute pointers for the V consecutive elements that this thread handles
    a_ptrs = x_ptr + g_row * stride_x0 + g_col_start + offs
    b_ptrs = x_ptr + g_row * stride_x0 + g_col_start + N + offs
    out_ptrs = out_ptr + g_row * stride_out0 + g_col_start + offs

    # Masks – for the given fixed 8192×8192 shape they are always true,
    # but keep them for generality and let the compiler prove it at JIT time.
    mask = (g_row < M) & ((g_col_start + offs) < N)

    a = tl.load(a_ptrs, mask=mask, other=0.0)
    b = tl.load(b_ptrs, mask=mask, other=0.0)

    silu_a = a * tl.sigmoid(a)
    out_val = silu_a * b

    tl.store(out_ptrs, out_val, mask=mask)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x:  float32 tensor of shape [8192, 8192]
    Returns:
        out: float32 tensor of shape [8192, 4096]
            out = silu(x[..., :4096]) * x[..., 4096:]
    """
    assert x.ndim == 2 and x.shape[1] % 2 == 0, "Input must be 2D with even last dim"
    M, D = x.shape
    N = D // 2
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # Tile and vectorization parameters.
    BLOCK_M = 8
    BLOCK_N = 128
    V = 4                     # elements per thread along columns
    N_THREADS_COL = BLOCK_N // V
    total_threads = BLOCK_M * N_THREADS_COL
    num_warps = total_threads // 32

    # Because 8192 % 8 == 0 and 4096 % 128 == 0, the grid is exact.
    grid = (M // BLOCK_M, N // BLOCK_N)

    swiglu_kernel[grid](
        x,
        out,
        M,
        N,
        x.stride(0),
        out.stride(0),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        V=V,
        num_warps=num_warps,
        num_stages=3,
    )
    return out