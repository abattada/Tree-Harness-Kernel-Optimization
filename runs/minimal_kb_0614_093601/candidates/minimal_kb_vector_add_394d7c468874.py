import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel: elementwise vector addition  (x + y)
# ---------------------------------------------------------------------------

@triton.jit
def vector_add_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements: int,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


# ---------------------------------------------------------------------------
# Public entry point – allocates output and launches the kernel.
# ---------------------------------------------------------------------------

def triton_run(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    x: float32 tensor of shape [16_000_000]
    y: float32 tensor of shape [16_000_000]
    returns: float32 tensor of shape [16_000_000]
    """
    assert x.device == y.device and x.dtype == y.dtype == torch.float32
    n_elements = x.numel()
    out = torch.empty_like(x)

    # Launch configuration
    BLOCK_SIZE = 2048                      # balance occupancy & coalescing
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    vector_add_kernel[grid](
        x, y, out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=4,
    )
    return out