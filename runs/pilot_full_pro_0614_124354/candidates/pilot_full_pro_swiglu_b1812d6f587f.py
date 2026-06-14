import torch
import triton
import triton.language as tl


@triton.jit
def swiglu_kernel(
    x_ptr, out_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_outm, stride_outn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # 2D grid: program_id(0) over rows, program_id(1) over columns
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # shape (BLOCK_M,)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # shape (BLOCK_N,)

    # Broadcast to a 2D tile for vectorised load/store
    offs_m = offs_m[:, None]      # (BLOCK_M, 1)
    offs_n = offs_n[None, :]      # (1, BLOCK_N)

    mask = (offs_m < M) & (offs_n < N)

    # a and b are contiguous halves of x along the last dimension
    a_ptrs = x_ptr + offs_m * stride_xm + offs_n * stride_xn
    b_ptrs = x_ptr + offs_m * stride_xm + (offs_n + N) * stride_xn
    out_ptrs = out_ptr + offs_m * stride_outm + offs_n * stride_outn

    a = tl.load(a_ptrs, mask=mask, other=0.0)
    b = tl.load(b_ptrs, mask=mask, other=0.0)

    # SwiGLU activation: silu(a) * b  =  a * sigmoid(a) * b
    silu_a = a * tl.sigmoid(a)
    out = silu_a * b

    tl.store(out_ptrs, out, mask=mask)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    Apply SwiGLU elementwise: split x into two halves a, b along dim=-1,
    compute silu(a) * b.
    Input:  (M, 2N) float32
    Output: (M, N)   float32
    """
    assert x.dtype == torch.float32
    M, N2 = x.shape
    N = N2 // 2
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    # Tuned tile sizes:
    # BLOCK_M=1 gives perfect memory coalescing (every warp loads contiguous
    # elements from a and b sequentially); BLOCK_N=1024 keeps tile size
    # moderate and the grid large (8192 x 4 = 32768 programs) for high SM
    # occupancy, while still fitting in registers with 8 warps.
    BLOCK_M = 1
    BLOCK_N = 1024

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    swiglu_kernel[grid](
        x, out,
        M, N,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=8, num_stages=2,
    )
    return out