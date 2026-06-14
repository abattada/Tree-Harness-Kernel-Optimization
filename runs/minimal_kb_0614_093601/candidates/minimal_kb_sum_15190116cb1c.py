import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# First reduction stage: each block sums a contiguous chunk of the input.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_kernel(
    x_ptr,                                          # input: shape (N,)
    partials_ptr,                                   # output partial sums: (num_blocks,)
    N: int,                                         # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    block_sum = tl.sum(x, axis=0)

    # Store the partial sum
    tl.store(partials_ptr + pid, block_sum)


# ---------------------------------------------------------------------------
# Second reduction stage: combine all partial sums into one scalar.
# This kernel uses a loop over blocks of the partials array, accumulating
# into a single float, which is then stored to the output scalar tensor.
# ---------------------------------------------------------------------------
@triton.jit
def combine_kernel(
    partials_ptr,                                   # partial sums: (num_partials,)
    out_ptr,                                        # scalar output
    num_partials: int,
    BLOCK_SIZE: tl.constexpr,
):
    # A single program accumulates all partial sums
    acc = 0.0
    start = 0
    # Grid‑stride loop: each iteration handles BLOCK_SIZE elements
    while start < num_partials:
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_partials
        vals = tl.load(partials_ptr + offsets, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)
        start += BLOCK_SIZE

    # Store final result to scalar output
    tl.store(out_ptr, acc)


# ---------------------------------------------------------------------------
# Public API: one‑dimensional sum reduction.
# ---------------------------------------------------------------------------
def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32
    assert x.dim() == 1, "Input must be 1‑dimensional"

    N = x.numel()

    # Tunable parameters (seed values – can be tuned later)
    BLOCK_SIZE_REDUCE = 8192      # 32K per block → 8K floats per block
    BLOCK_SIZE_COMBINE = 1024    # 4K floats per combine tile
    NUM_WARPS = 4
    NUM_STAGES = 4

    num_blocks = (N + BLOCK_SIZE_REDUCE - 1) // BLOCK_SIZE_REDUCE
    partials = torch.empty(num_blocks, dtype=torch.float32, device=x.device)

    # First kernel: reduce each block
    grid_reduce = (num_blocks,)
    reduce_kernel[grid_reduce](
        x, partials, N,
        BLOCK_SIZE=BLOCK_SIZE_REDUCE,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )

    # Output scalar tensor
    out = torch.empty((), dtype=torch.float32, device=x.device)

    # Second kernel: combine partial sums (single program)
    grid_combine = (1,)
    combine_kernel[grid_combine](
        partials, out, num_blocks,
        BLOCK_SIZE=BLOCK_SIZE_COMBINE,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )

    return out