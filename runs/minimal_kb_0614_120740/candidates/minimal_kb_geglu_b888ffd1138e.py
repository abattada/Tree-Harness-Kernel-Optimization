import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ---------------------------------------------------------------------------
# Triton kernel: GeGLU (gated GELU with tanh approximation)
# Input:  x [M, K] with K even, K = 8192
# Output: out [M, K//2] = [M, 4096]
#
# Persistent-kernel formulation: launches only as many programs as there are
# SMs, each program processes a chunk of rows in a grid-stride loop. This
# reduces launch overhead and improves load balancing.
# ---------------------------------------------------------------------------

@triton.jit
def geglu_kernel(
    x_ptr,                # pointer to input (2D flattened)
    out_ptr,              # pointer to output (2D flattened)
    stride_x,             # row stride of x (in elements) = K
    stride_out,           # row stride of out (in elements) = N
    M: tl.int32,          # total number of rows
    BLOCK_SIZE: tl.constexpr,   # = N (output dimension)
):
    # Static assertion: we assume full rows (no padding)
    tl.static_assert(BLOCK_SIZE == 4096, "BLOCK_SIZE must be 4096 for this shape")

    # ---- 1. persistent grid-stride loop over rows ------------------------
    row = tl.program_id(0)
    num_programs = tl.num_programs(0)
    col_offs = tl.arange(0, BLOCK_SIZE)

    while row < M:
        # Row pointers
        x_row = x_ptr + row * stride_x
        out_row = out_ptr + row * stride_out

        # Load a (first half) and b (second half) with evict_first hint
        a = tl.load(x_row + col_offs, eviction_policy='evict_first')
        b = tl.load(x_row + col_offs + BLOCK_SIZE, eviction_policy='evict_first')

        # ---- 2. compute GELU_tanh(a) -------------------------------------
        # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        sqrt2_over_pi = 0.7978845608028654
        c = 0.044715

        a3 = a * a * a
        inner = sqrt2_over_pi * (a + c * a3)
        tanh_val = libdevice.tanh(inner)
        gelu_a = 0.5 * a * (1.0 + tanh_val)

        # ---- 3. gating ----------------------------------------------------
        result = gelu_a * b

        # ---- 4. store result (evict_first) --------------------------------
        tl.store(out_row + col_offs, result, eviction_policy='evict_first')

        row += num_programs


# ---------------------------------------------------------------------------
# Public entry point – allocates output and launches the kernel.
# ---------------------------------------------------------------------------

def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    x: float32 tensor of shape [M, K]  (K even)
    returns: float32 tensor of shape [M, K//2]
    """
    M, K = x.shape
    N = K // 2                          # output feature dimension

    # Allocate output
    out = torch.empty(M, N, device=x.device, dtype=torch.float32)

    # Persistent kernel: use as many programs as SMs (or a sensible max)
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    num_programs = min(M, sm_count)     # cap at number of rows
    grid = (num_programs,)

    BLOCK_SIZE = N                      # 4096 – full output row

    geglu_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        M,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=4,
    )

    return out