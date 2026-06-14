import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def _rms_norm_kernel(
    x_ptr,
    y_ptr,
    N_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    EPS: tl.constexpr,
):
    """RMS normalization kernel: each block processes one row."""
    row_id = tl.program_id(0)

    # Shared memory to hold the whole row (16 KB for 4096 f32)
    smem = tl.static_shared_memory(shape=(N_COLS,), dtype=tl.float32)
    smem_ptr = smem.data

    # Partial sum accumulator per thread
    partials = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    num_tiles = N_COLS // BLOCK_SIZE  # integer since BLOCK_SIZE divides N_COLS

    # First pass: read row, accumulate squares, cache in shared memory
    for tile in range(num_tiles):
        offs = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        vals = tl.load(x_ptr + row_id * N_COLS + offs)
        tl.store(smem_ptr + offs, vals)
        partials += vals * vals

    # Block‑wide sum of squares → mean of squares
    sum_sq = tl.sum(partials)
    mean_sq = sum_sq / N_COLS

    # Reciprocal square root with epsilon
    rsqrt_val = tl.rsqrt(mean_sq + EPS)

    # Second pass: scale elements and write output
    for tile in range(num_tiles):
        offs = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        scaled = tl.load(smem_ptr + offs) * rsqrt_val
        tl.store(y_ptr + row_id * N_COLS + offs, scaled)


def triton_run(x: torch.Tensor):
    """RMS normalization: x * rsqrt(mean(x^2, dim=-1) + 1e-5)."""
    N_ROWS, N_COLS = x.shape
    assert N_COLS == 4096, "This kernel expects N_COLS=4096"

    out = torch.empty_like(x)

    # Chosen BLOCK_SIZE divides N_COLS exactly; num_warps matches BLOCK_SIZE/32
    BLOCK_SIZE = 512
    grid = (N_ROWS,)

    _rms_norm_kernel[grid](
        x,
        out,
        N_COLS=N_COLS,
        BLOCK_SIZE=BLOCK_SIZE,
        EPS=1e-5,
        num_warps=16,
    )
    return out