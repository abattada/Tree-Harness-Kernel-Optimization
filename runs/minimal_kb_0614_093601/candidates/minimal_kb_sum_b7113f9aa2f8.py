import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# First reduction stage: each program sums a contiguous chunk of the input.
# Uses a grid-stride loop within the chunk to reduce launch count.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_kernel(
    x_ptr,                                          # input: shape (N,)
    partials_ptr,                                   # output partial sums: (num_programs,)
    N: int,                                         # total number of elements
    BLOCK_SIZE: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,                       # elements per program
):
    pid = tl.program_id(0)
    # Starting offset for this program
    start = pid * CHUNK_SIZE
    # End of chunk (exclusive)
    end = tl.minimum(start + CHUNK_SIZE, N)

    acc = 0.0
    # Loop over blocks inside the chunk
    for block_start in range(start, end, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < end
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        acc += tl.sum(x, axis=0)

    # Store partial sum
    tl.store(partials_ptr + pid, acc)


# ---------------------------------------------------------------------------
# Second reduction stage: combine all partial sums into a single scalar.
# Uses a single block; the number of partials is small (< 1024), so we
# directly sum with a grid-stride loop.
# ---------------------------------------------------------------------------
@triton.jit
def combine_kernel(
    partials_ptr,                                   # partial sums: (num_partials,)
    out_ptr,                                        # scalar output
    num_partials: int,
    BLOCK_SIZE: tl.constexpr,
):
    acc = 0.0
    start = 0
    while start < num_partials:
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_partials
        vals = tl.load(partials_ptr + offsets, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)
        start += BLOCK_SIZE

    tl.store(out_ptr, acc)


# ---------------------------------------------------------------------------
# Public API: one‑dimensional sum reduction.
# ---------------------------------------------------------------------------
def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32
    assert x.dim() == 1, "Input must be 1‑dimensional"

    N = x.numel()

    # Tunable parameters – chosen for good occupancy and bandwidth
    BLOCK_SIZE_REDUCE = 1024         # 4KB per load
    CHUNK_SIZE = 65536               # 64K elements per program → 512 programs for 32M
    BLOCK_SIZE_COMBINE = 512         # 2KB per load (enough for 512 partials)
    NUM_WARPS = 4
    NUM_STAGES = 4

    num_programs = (N + CHUNK_SIZE - 1) // CHUNK_SIZE
    partials = torch.empty(num_programs, dtype=torch.float32, device=x.device)

    # First kernel: each program sums a chunk
    grid_reduce = (num_programs,)
    reduce_kernel[grid_reduce](
        x, partials, N,
        BLOCK_SIZE=BLOCK_SIZE_REDUCE,
        CHUNK_SIZE=CHUNK_SIZE,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )

    # Output scalar tensor
    out = torch.empty((), dtype=torch.float32, device=x.device)

    # Second kernel: combine partial sums (single program)
    grid_combine = (1,)
    combine_kernel[grid_combine](
        partials, out, num_programs,
        BLOCK_SIZE=BLOCK_SIZE_COMBINE,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )

    return out