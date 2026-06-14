import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel: elementwise exponential (vector_exp)
# Works on a contiguous 1-D tensor of float32.
# One program per BLOCK_SIZE elements.
# ---------------------------------------------------------------------------

@triton.jit
def exp_kernel(
    x_ptr,           # input pointer (f32, 1D)
    out_ptr,         # output pointer (f32, 1D)
    n_elements,      # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # 1. Compute the block-aligned index range for this program
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements                      # guard for the last block
    # 2. Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # 3. Compute exp(x)
    y = tl.exp(x)
    # 4. Store output
    tl.store(out_ptr + offsets, y, mask=mask)


# ---------------------------------------------------------------------------
# Public entry point – allocates output and launches the kernel.
# ---------------------------------------------------------------------------

def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    x:  float32 tensor of shape [16M]  (1-D, any length)
    returns: float32 tensor of the same shape, elementwise exp(x)
    """
    n_elements = x.numel()
    # Allocate output
    out = torch.empty_like(x)

    # Launch configuration (tunable)
    BLOCK_SIZE = 4096               # covers a large contiguous chunk
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    exp_kernel[grid](
        x, out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,                # moderate warp count – balanced
        num_stages=4,               # standard pipelining depth
    )
    return out