import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel: elementwise vector addition (x + y)
# Input:  1D tensors x, y of length N
# Output: 1D tensor out of length N  (x + y)
# ---------------------------------------------------------------------------

@triton.jit
def vector_add_kernel(
    x_ptr,          # pointer to input x
    y_ptr,          # pointer to input y
    out_ptr,        # pointer to output out
    N,              # total number of elements
    BLOCK_SIZE: tl.constexpr,   # number of elements per block
):
    # ---- 1. block index (convert to int64 for safe large-N computation) ----
    block_id = tl.program_id(0).to(tl.int64)

    # ---- 2. element offsets for this block --------------------------------
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # ---- 3. mask for the (potentially partial) final block ----------------
    mask = offsets < N

    # ---- 4. load x and y --------------------------------------------------
    x_val = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y_val = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # ---- 5. compute x + y -------------------------------------------------
    out_val = x_val + y_val

    # ---- 6. store result --------------------------------------------------
    tl.store(out_ptr + offsets, out_val, mask=mask)


# ---------------------------------------------------------------------------
# Public entry point – allocates output and launches the kernel.
# ---------------------------------------------------------------------------

def triton_run(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    x: float32 tensor of shape [N]  (N = 16M)
    y: float32 tensor of shape [N]
    returns: float32 tensor of shape [N]  (x + y)
    """
    # Input validation (optional, but good practice)
    assert x.is_cuda and y.is_cuda
    assert x.dtype == torch.float32 and y.dtype == torch.float32
    assert x.shape == y.shape
    N = x.shape[0]

    # Allocate output
    out = torch.empty(N, device=x.device, dtype=torch.float32)

    # Launch configuration
    BLOCK_SIZE = 4096                # covers a contiguous chunk of 4096 elements (16 KB)
    grid = (triton.cdiv(N, BLOCK_SIZE),)   # number of blocks to cover all elements

    vector_add_kernel[grid](
        x, y, out,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=4,
    )
    return out