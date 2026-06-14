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
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # row and column offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # broadcast to 2D tile
    offs_m = offs_m[:, None]
    offs_n = offs_n[None, :]
    
    # mask for out-of-bounds elements
    mask = (offs_m < M) & (offs_n < N)
    
    # point to a = x[:, :N], b = x[:, N:]
    a_ptrs = x_ptr + offs_m * stride_xm + offs_n * stride_xn
    b_ptrs = x_ptr + offs_m * stride_xm + (offs_n + N) * stride_xn
    out_ptrs = out_ptr + offs_m * stride_outm + offs_n * stride_outn
    
    a = tl.load(a_ptrs, mask=mask, other=0.0)
    b = tl.load(b_ptrs, mask=mask, other=0.0)
    
    # SwiGLU: silu(a) * b = a * sigmoid(a) * b
    silu_a = a * tl.sigmoid(a)
    out = silu_a * b
    
    tl.store(out_ptrs, out, mask=mask)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float32
    M, N2 = x.shape
    N = N2 // 2  # output column count
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    
    # Tile sizes – obvious tuning knobs to sweep later
    BLOCK_M = 4
    BLOCK_N = 256
    
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