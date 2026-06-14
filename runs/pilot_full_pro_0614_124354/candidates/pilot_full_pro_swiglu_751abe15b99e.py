import torch
import triton
import triton.language as tl

# ------------------------------------------------------------------------------
# Kernel
# ------------------------------------------------------------------------------
@triton.jit
def _swiglu_kernel(
    x_ptr,
    out_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Compute SwiGLU: output = silu(a) * b
    where a = x[:, :N//2], b = x[:, N//2:]
    N = 8192, output N_out = 4096 (half)
    x and out are row-major.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row and column offsets for the output tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)          # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)          # [BLOCK_N]

    # Input is (8192, 8192).  Half dimension = 4096.
    # Pointer to a (first half) and b (second half)
    a_ptr = x_ptr + offs_m[:, None] * 8192 + offs_n[None, :]
    b_ptr = x_ptr + offs_m[:, None] * 8192 + (offs_n[None, :] + 4096)

    a = tl.load(a_ptr)       # [BLOCK_M, BLOCK_N]
    b = tl.load(b_ptr)

    silu_a = a * tl.sigmoid(a)
    o = silu_a * b

    # Output has shape (8192, 4096)
    out_ptr = out_ptr + offs_m[:, None] * 4096 + offs_n[None, :]
    tl.store(out_ptr, o)


# ------------------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------------------
def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    x: float32 tensor of shape [8192, 8192]
    returns: float32 tensor of shape [8192, 4096]
    """
    assert x.is_cuda and x.dtype == torch.float32
    assert x.shape == (8192, 8192)
    if not x.is_contiguous():
        x = x.contiguous()

    M, N = x.shape
    N_out = N // 2  # 4096

    # Choose block sizes that divide the dimensions evenly and give
    # at most 512 threads per block (16 warps)
    BLOCK_M = 4
    BLOCK_N = 128

    grid_m = triton.cdiv(M, BLOCK_M)
    grid_n = triton.cdiv(N_out, BLOCK_N)

    out = torch.empty((M, N_out), device=x.device, dtype=x.dtype)

    _swiglu_kernel[(grid_m, grid_n)](
        x,
        out,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=16,
        num_stages=4,
    )
    return out