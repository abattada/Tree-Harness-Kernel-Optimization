import torch
import triton
import triton.language as tl


@triton.jit
def swiglu_kernel(
    x_ptr,
    out_ptr,
    stride_x0,
    stride_out0,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Computes SwiGLU output = silu(a) * b where a = x[:,:N], b = x[:,N:2N].
    M and N are known constexpr shapes that are exact multiples of the
    tiling dimensions, so there is no need for boundary masks.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 2‑D tile index ranges (no mask needed – perfectly bounded)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Broadcast for 2‑D pointer arithmetic
    rows_2d = rows[:, None]
    cols_2d = cols[None, :]

    # Pointers for a (left half), b (right half), and output
    a_ptrs = x_ptr + rows_2d * stride_x0 + cols_2d
    b_ptrs = x_ptr + rows_2d * stride_x0 + (cols_2d + N)
    out_ptrs = out_ptr + rows_2d * stride_out0 + cols_2d

    # Load with no mask (indices guaranteed in bounds)
    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)

    # SiLU activation
    silu_a = a * tl.sigmoid(a)
    out_val = silu_a * b

    tl.store(out_ptrs, out_val)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    SwiGLU for a fixed 8192×8192 f32 input, returning 8192×4096 f32.
    a, b = x.chunk(2, dim=-1); out = silu(a) * b
    """
    # Hard constraint from the problem statement
    assert x.shape == (8192, 8192), "Input must be exactly [8192, 8192] f32"
    M, D = 8192, 8192
    N = D // 2  # 4096
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)

    # Tuned tile shape: perfectly divides the fixed problem sizes, maximising
    # occupancy (16 warps, 512 threads) while keeping per‑thread work modest.
    BLOCK_M = 16     # 8192 / 16 = 512
    BLOCK_N = 512    # 4096 / 512 = 8
    grid = (M // BLOCK_M, N // BLOCK_N)

    swiglu_kernel[grid](
        x,
        out,
        x.stride(0),   # = 8192, contiguous
        out.stride(0), # = 4096, contiguous
        M=M,
        N=N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=16,
        num_stages=2,
    )
    return out