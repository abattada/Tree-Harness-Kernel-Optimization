import torch
import triton
import triton.language as tl

# Reduction parameters
BLOCK_SIZE = 1024      # number of elements per tile
NUM_WARPS = 4          # warps per block
NUM_STAGES = 4         # software pipeline stages (not heavily used here)
NUM_PARTIALS = 256     # number of blocks in first stage (grid size)

@triton.jit
def sum_partial_kernel(
    x_ptr,          # input: [N]
    partials_ptr,   # output: [NUM_PARTIALS] (partial sums)
    N: int,         # total number of elements
    BLOCK_SIZE: tl.constexpr,
    NUM_PARTIALS: tl.constexpr,
):
    pid = tl.program_id(0)

    # Each block processes a contiguous chunk of the input
    chunk_size = tl.cdiv(N, NUM_PARTIALS)
    block_start = pid * chunk_size
    block_end = tl.minimum(block_start + chunk_size, N)

    total = 0.0
    for col_start in range(block_start, block_end, BLOCK_SIZE):
        offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < block_end
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        total += tl.sum(x, axis=0)

    # Write partial sum
    tl.store(partials_ptr + pid, total)


@triton.jit
def combine_reduce_kernel(
    partials_ptr,   # input: [NUM_PARTIALS]
    output_ptr,     # output: scalar
    NUM_PARTIALS: int,
    BLOCK_SIZE: tl.constexpr,
):
    # Single block sums all partials
    total = 0.0
    for start in range(0, NUM_PARTIALS, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < NUM_PARTIALS
        vals = tl.load(partials_ptr + offsets, mask=mask, other=0.0)
        total += tl.sum(vals, axis=0)

    # Thread 0 writes the final scalar
    if tl.program_id(0) == 0:
        tl.store(output_ptr, total)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32
    N = x.numel()

    # Intermediate buffer for partial sums
    partials = torch.empty(NUM_PARTIALS, dtype=torch.float32, device=x.device)
    # Output scalar tensor
    out = torch.empty((), dtype=torch.float32, device=x.device)

    # First stage: compute partial sums across the input
    grid_stage1 = (NUM_PARTIALS,)
    sum_partial_kernel[grid_stage1](
        x,
        partials,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_PARTIALS=NUM_PARTIALS,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )

    # Second stage: combine partials into final scalar
    grid_stage2 = (1,)
    combine_reduce_kernel[grid_stage2](
        partials,
        out,
        NUM_PARTIALS,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )

    return out