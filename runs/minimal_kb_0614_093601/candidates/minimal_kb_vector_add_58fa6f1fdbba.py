import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel: vector_add
# Inputs: x, y  (1D float32 tensors of length N)
# Output: out = x + y
# ---------------------------------------------------------------------------

@triton.jit
def add_kernel(
    x_ptr, y_ptr, out_ptr,
    stride_x, stride_y, stride_out,   # stride in elements (always 1 for contiguous)
    N,                                 # total number of elements
    BLOCK_SIZE: tl.constexpr,          # number of elements per program
):
    # ---- 1. global element offset for this program --------------------------
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # ---- 2. pointers for this chunk -----------------------------------------
    x_ptrs = x_ptr + offsets
    y_ptrs = y_ptr + offsets
    out_ptrs = out_ptr + offsets

    # ---- 3. load, add, store ------------------------------------------------
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    y = tl.load(y_ptrs, mask=mask, other=0.0)
    z = x + y
    tl.store(out_ptrs, z, mask=mask)


# ---------------------------------------------------------------------------
# Public entry point – allocates output and launches the kernel.
# ---------------------------------------------------------------------------

def triton_run(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    x, y: float32 tensors of shape [16,777,216] (contiguous)
    returns: float32 tensor of same shape, containing x+y
    """
    assert x.is_cuda and y.is_cuda
    assert x.shape == y.shape and x.numel() == y.numel()
    N = x.numel()

    # Allocate output
    out = torch.empty_like(x)

    # Launch configuration
    BLOCK_SIZE = 4096                     # 2^12, divides 16M exactly
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)   # ceil(N/BLOCK_SIZE)

    add_kernel[grid](
        x, y, out,
        x.stride(0), y.stride(0), out.stride(0),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=4,
    )
    return out