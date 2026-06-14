import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel: elementwise exp on a 1-D tensor
# Each program processes a contiguous block of BLOCK_SIZE elements.
# ---------------------------------------------------------------------------

@triton.jit
def vector_exp_kernel(
    x_ptr,          # pointer to input tensor (1D)
    out_ptr,        # pointer to output tensor (1D)
    N,              # total number of elements
    BLOCK_SIZE: tl.constexpr,   # number of elements per program (must be a power of two)
):
    # Compute the starting index of this program
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Compute column offsets within the block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    # Global indices
    indices = block_start + col_offsets
    # Mask to handle the last non-full block
    mask = indices < N

    # Load input (with mask; out-of-bound elements set to 0.0)
    x = tl.load(x_ptr + indices, mask=mask, other=0.0)

    # Compute exp elementwise
    y = tl.exp(x)

    # Store output
    tl.store(out_ptr + indices, y, mask=mask)


# ---------------------------------------------------------------------------
# Public entry point – allocates output and launches the kernel.
# ---------------------------------------------------------------------------

def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    x: float32 tensor of shape [N] (1-D)
    returns: float32 tensor of shape [N] = exp(x)
    """
    assert x.is_cuda and x.dtype == torch.float32
    N = x.numel()
    out = torch.empty_like(x)

    # Launch configuration
    BLOCK_SIZE = 4096                      # good balance for 1D elementwise
    grid = (triton.cdiv(N, BLOCK_SIZE),)   # one program per block

    vector_exp_kernel[grid](
        x, out, N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=4,
    )
    return out