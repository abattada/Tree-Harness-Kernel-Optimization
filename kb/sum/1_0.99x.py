import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Persistent first-stage reduction: each program reduces a contiguous chunk
# of the input and writes one partial sum.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_kernel(
    x_ptr,                                          # input: (N,)
    partials_ptr,                                   # output partial sums: (num_programs,)
    N: int,                                         # total number of elements
    BLOCK_SIZE: tl.constexpr,
    num_programs: tl.constexpr,
):
    pid = tl.program_id(0)

    # Compute the contiguous chunk assigned to this program
    chunk_start = pid * (N // num_programs)
    chunk_end = (pid + 1) * (N // num_programs) if pid < num_programs - 1 else N

    acc = 0.0
    # Loop over the chunk with BLOCK_SIZE stride
    for i in range(chunk_start, chunk_end, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < chunk_end
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        acc += tl.sum(x, axis=0)

    tl.store(partials_ptr + pid, acc)


# ---------------------------------------------------------------------------
# Second stage: combine all partial sums into one scalar.
# Since num_programs is small (256), a single block suffices.
# ---------------------------------------------------------------------------
@triton.jit
def combine_kernel(
    partials_ptr,                                   # partial sums: (num_programs,)
    out_ptr,                                        # scalar output
    num_programs: int,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_programs
    vals = tl.load(partials_ptr + offsets, mask=mask, other=0.0)
    result = tl.sum(vals, axis=0)
    tl.store(out_ptr, result)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32
    N = x.numel()

    # Tunable parameters
    BLOCK_SIZE_REDUCE = 4096      # 16 KB per load block (good coalescing)
    BLOCK_SIZE_COMBINE = 256      # power of two covering the number of programs
    NUM_WARPS_REDUCE = 8          # more warps to hide latency in reduction
    NUM_STAGES_REDUCE = 4
    NUM_WARPS_COMBINE = 1
    NUM_STAGES_COMBINE = 2

    # Number of persistent programs: limit to 256 to avoid excessive launch
    # overhead while keeping each chunk large enough for good efficiency.
    num_programs = 256
    # If the input is smaller than 256 * BLOCK_SIZE_REDUCE, reduce programs accordingly
    # to avoid idle threads (masked loads are cheap but still suboptimal).
    max_blocks = (N + BLOCK_SIZE_REDUCE - 1) // BLOCK_SIZE_REDUCE
    if num_programs > max_blocks:
        num_programs = max_blocks
    num_programs = max(num_programs, 1)  # at least one program

    partials = torch.empty(num_programs, dtype=torch.float32, device=x.device)
    grid_reduce = (num_programs,)
    reduce_kernel[grid_reduce](
        x, partials, N,
        BLOCK_SIZE=BLOCK_SIZE_REDUCE,
        num_programs=num_programs,
        num_warps=NUM_WARPS_REDUCE,
        num_stages=NUM_STAGES_REDUCE,
    )

    out = torch.empty((), dtype=torch.float32, device=x.device)
    grid_combine = (1,)
    combine_kernel[grid_combine](
        partials, out, num_programs,
        BLOCK_SIZE=BLOCK_SIZE_COMBINE,
        num_warps=NUM_WARPS_COMBINE,
        num_stages=NUM_STAGES_COMBINE,
    )

    return out