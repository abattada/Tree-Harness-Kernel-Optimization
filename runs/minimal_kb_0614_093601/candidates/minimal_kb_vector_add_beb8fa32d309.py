import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel: vector_add (elementwise x + y)
# Inputs:  x, y  [N] float32
# Output:  out   [N] float32
# Each program processes a contiguous block of BLOCK_SIZE elements.
# ---------------------------------------------------------------------------

@triton.jit
def vector_add_kernel(
    x_ptr,  # pointer to first input
    y_ptr,  # pointer to second input
    out_ptr,  # pointer to output
    n_elements,  # total number of elements
    BLOCK_SIZE: tl.constexpr,  # number of elements per program (power of two)
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Coalesced loads
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Elementwise addition
    result = x + y

    # Coalesced store
    tl.store(out_ptr + offsets, result, mask=mask)


# ---------------------------------------------------------------------------
# Public entry point – allocates output and launches the kernel.
# ---------------------------------------------------------------------------

def triton_run(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    x, y: float32 tensors of shape [16M] (16,777,216 elements)
    returns: float32 tensor of same shape (x + y)
    """
    n = x.numel()  # 16,777,216
    out = torch.empty_like(x)

    # Launch configuration – tune BLOCK_SIZE, num_warps, num_stages for performance
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    vector_add_kernel[grid](
        x, y, out,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=4,
    )
    return out