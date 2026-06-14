import torch
import triton
import triton.language as tl

# Tunable parameters for the reduction
BLOCK_SIZE = 1024          # Size of contiguous chunk loaded per iteration per program (first stage)
FIRST_STAGE_PROGRAMS = 1024  # Number of partial sums (grid size of first kernel)
BLOCK_REDUCE = 1024        # Block size for the final reduction kernel (must be power of two >= FIRST_STAGE_PROGRAMS)
NUM_WARPS_FIRST = 4
NUM_STAGES_FIRST = 4
NUM_WARPS_REDUCE = 4
NUM_STAGES_REDUCE = 4

@triton.jit
def sum_stage1_kernel(
    x_ptr,            # input: [n_elements]
    partials_ptr,     # output partial sums: [FIRST_STAGE_PROGRAMS]
    n_elements: int,  # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    # Each program iterates over the input with stride FIRST_STAGE_PROGRAMS * BLOCK_SIZE
    grid_stride = FIRST_STAGE_PROGRAMS * BLOCK_SIZE
    acc = tl.zeros([1], dtype=tl.float32)  # scalar accumulator

    offsets_base = pid * BLOCK_SIZE
    for start in range(offsets_base, n_elements, grid_stride):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        # Guard against invalid elements if the last block is partial
        # tl.sum handles masking automatically: zeros out masked elements
        block_sum = tl.sum(x, axis=0)
        acc = acc + block_sum

    # Write the partial sum to the output array
    tl.store(partials_ptr + pid, acc)

@triton.jit
def sum_stage2_kernel(
    partials_ptr,       # input: partial sums array of length FIRST_STAGE_PROGRAMS
    out_ptr,            # output scalar (0-dim tensor)
    BLOCK_REDUCE: tl.constexpr,
):
    pid = tl.program_id(0)  # only one program
    offsets = tl.arange(0, BLOCK_REDUCE)
    mask = offsets < FIRST_STAGE_PROGRAMS
    partials = tl.load(partials_ptr + offsets, mask=mask, other=0.0)
    total = tl.sum(partials, axis=0)
    # Store the scalar; out_ptr is a pointer to a single element
    tl.store(out_ptr, total)

def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32
    n_elements = x.numel()
    # Allocate output scalar tensor
    out = torch.empty((), dtype=torch.float32, device=x.device)
    # Allocate partial sums array (GPU memory)
    partials = torch.empty(FIRST_STAGE_PROGRAMS, dtype=torch.float32, device=x.device)

    # First stage: reduce to partial sums
    grid1 = (FIRST_STAGE_PROGRAMS,)
    sum_stage1_kernel[grid1](
        x, partials, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=NUM_WARPS_FIRST,
        num_stages=NUM_STAGES_FIRST,
    )

    # Second stage: combine partial sums into scalar
    grid2 = (1,)
    sum_stage2_kernel[grid2](
        partials, out,
        BLOCK_REDUCE=BLOCK_REDUCE,
        num_warps=NUM_WARPS_REDUCE,
        num_stages=NUM_STAGES_REDUCE,
    )

    return out