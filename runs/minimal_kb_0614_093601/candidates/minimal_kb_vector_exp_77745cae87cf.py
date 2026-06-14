import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel: elementwise exponential (exp)
# ---------------------------------------------------------------------------

@triton.jit
def exp_kernel(
    x_ptr,          # pointer to input tensor (1D)
    out_ptr,        # pointer to output tensor (1D)
    n_elements,     # total number of elements
    BLOCK_SIZE: tl.constexpr,   # number of elements per block
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Use hints for better vectorization
    offsets = tl.multiple_of(offsets, 8)   # assume 32‑byte alignment per thread group? harmless hint
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0, eviction_policy='evict_first')
    out = tl.exp(x)
    tl.store(out_ptr + offsets, out, mask=mask, eviction_policy='evict_first')


# ---------------------------------------------------------------------------
# Public entry point – allocates output and launches the kernel.
# ---------------------------------------------------------------------------

def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    x: float32 tensor of shape [16M] (1D)
    returns: float32 tensor of the same shape, elementwise exp(x)
    """
    assert x.is_cuda and x.dtype == torch.float32
    n = x.numel()                                 # 16,777,216 exactly
    out = torch.empty_like(x)                     # allocate output

    # Launch configuration – tuned for RTX 5090 (Blackwell)
    BLOCK_SIZE = 8192                             # larger block keeps threads busy
    grid = (triton.cdiv(n, BLOCK_SIZE),)          # 2048 programs

    exp_kernel[grid](
        x, out,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,       # increased occupancy
        num_stages=4,      # default pipeline depth
    )
    return out