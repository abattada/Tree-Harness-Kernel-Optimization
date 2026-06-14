import torch
import triton
import triton.language as tl

@triton.jit
def exp_kernel(
    x_ptr,                # pointer to input tensor
    out_ptr,              # pointer to output tensor
    n_elements,           # total number of elements
    BLOCK_SIZE: tl.constexpr,   # number of elements per program
):
    pid = tl.program_id(0)            # block index in 1D grid
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input, compute exp, store output
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.exp(x)
    tl.store(out_ptr + offsets, y, mask=mask)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    Compute elementwise exp(x) for a 1D float32 tensor.
    """
    assert x.is_cuda and x.dtype == torch.float32 and x.dim() == 1
    n_elements = x.numel()

    # Allocate output
    out = torch.empty_like(x)

    # Launch configuration – tuned for the test size (16M)
    # BLOCK_SIZE = 4096 ensures 4096 programs, good occupancy.
    BLOCK_SIZE = 4096
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    exp_kernel[grid](
        x, out, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=4,
    )
    return out