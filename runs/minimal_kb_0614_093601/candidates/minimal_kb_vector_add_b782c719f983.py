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
    x_ptr,             # pointer to first input
    y_ptr,             # pointer to second input
    out_ptr,           # pointer to output
    n_elements,        # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Use alignment hints to enable vectorized loads/stores
    x_ptr = tl.max_contiguous(tl.multiple_of(x_ptr, BLOCK_SIZE), BLOCK_SIZE)
    y_ptr = tl.max_contiguous(tl.multiple_of(y_ptr, BLOCK_SIZE), BLOCK_SIZE)
    out_ptr = tl.max_contiguous(tl.multiple_of(out_ptr, BLOCK_SIZE), BLOCK_SIZE)

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    result = x + y

    tl.store(out_ptr + offsets, result, mask=mask)


# ---------------------------------------------------------------------------
# Public entry point – allocates output and launches the kernel.
# ---------------------------------------------------------------------------

def triton_run(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    x, y: float32 tensors of shape [16M] (16,777,216 elements)
    returns: float32 tensor of same shape (x + y)
    """
    n = x.numel()
    out = torch.empty_like(x)

    # Tuned launch configuration
    BLOCK_SIZE = 2048   # can be 1024, 2048, 4096; try larger block for fewer programs
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    vector_add_kernel[grid](
        x, y, out,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,        # more warps to hide memory latency
        num_stages=4,       # keep as before
    )
    return out