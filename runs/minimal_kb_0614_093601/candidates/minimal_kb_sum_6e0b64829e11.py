import torch
import triton
import triton.language as tl

# Tunable parameters (could be moved to outer scope for tuning)
BLOCK_SIZE_STAGE1 = 1024        # Must divide 32M (33554432) evenly
NUM_WARPS_STAGE1 = 4
NUM_STAGES_STAGE1 = 4
BLOCK_SIZE_STAGE2 = 1024        # Number of threads for the final reduction block
NUM_WARPS_STAGE2 = 8
NUM_STAGES_STAGE2 = 4


@triton.jit
def stage1_kernel(
    x_ptr,          # input [n_elements]
    partials_ptr,   # output [n_partials]
    n_elements: int,  # total number of elements (constexpr 33554432)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)  # block index along elements
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    # n_elements is a multiple of BLOCK_SIZE, so no masking needed
    x = tl.load(x_ptr + offsets)
    block_sum = tl.sum(x, axis=0).to(tl.float32)
    tl.store(partials_ptr + pid, block_sum)


@triton.jit
def stage2_kernel(
    partials_ptr,    # input [n_partials]
    out_ptr,         # output scalar (0-dim)
    n_partials: int, # number of partials (constexpr)
    BLOCK_SIZE: tl.constexpr,
):
    # Single block – program id is 0
    pid = tl.program_id(0)
    tid = tl.arange(0, BLOCK_SIZE)  # 1D thread indices

    # Each thread processes multiple partials
    # n_partials is a multiple of BLOCK_SIZE
    elems_per_thread = n_partials // BLOCK_SIZE
    my_sum = 0.0
    for i in range(elems_per_thread):
        idx = i * BLOCK_SIZE + tid
        v = tl.load(partials_ptr + idx)
        my_sum += v

    # Shared memory for tree reduction
    # Allocate BLOCK_SIZE floats on the stack (shared memory is automatic)
    scratch = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    # Write per-thread sum to shared memory
    scratch = tl.where(tid < BLOCK_SIZE, my_sum, scratch)
    # Make sure all threads have written
    tl.debug_barrier()  # Not needed in Triton but good practice

    # Tree reduction
    stride = BLOCK_SIZE // 2
    while stride > 0:
        # If the current thread is within the active half, do a load + add
        # We use tl.where to avoid predicated code divergence
        right = tl.load(scratch + tid + stride, mask=tid + stride < BLOCK_SIZE, other=0.0)
        left = tl.load(scratch + tid, mask=tid < BLOCK_SIZE, other=0.0)
        new_val = left + right
        # Store only for threads that are still active (tid < stride)
        tl.store(scratch + tid, new_val, mask=tid < stride)
        # Synchronize
        # In Triton, blocks are executed by one warp group, but we still need a barrier
        # Use tl.debug_barrier() to ensure correct ordering
        tl.debug_barrier()
        stride //= 2

    # Thread 0 writes the final sum
    final_sum = tl.load(scratch + 0)
    tl.store(out_ptr, final_sum)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32
    n_elements = x.numel()
    # Ensure n_elements is a multiple of BLOCK_SIZE_STAGE1 (true for 32M)
    assert n_elements % BLOCK_SIZE_STAGE1 == 0, \
        f"n_elements={n_elements} not divisible by BLOCK_SIZE_STAGE1={BLOCK_SIZE_STAGE1}"

    n_partials = n_elements // BLOCK_SIZE_STAGE1

    # Allocate output tensor (scalar, 0-dim)
    out = torch.empty([], dtype=torch.float32, device=x.device)

    # Stage 1: compute partial sums
    partials = torch.empty(n_partials, dtype=torch.float32, device=x.device)
    grid_stage1 = (n_partials,)
    stage1_kernel[grid_stage1](
        x, partials,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE_STAGE1,
        num_warps=NUM_WARPS_STAGE1,
        num_stages=NUM_STAGES_STAGE1,
    )

    # Stage 2: reduce partial sums to final scalar
    # Use a single block with BLOCK_SIZE_STAGE2 threads
    grid_stage2 = (1,)
    stage2_kernel[grid_stage2](
        partials, out,
        n_partials=n_partials,
        BLOCK_SIZE=BLOCK_SIZE_STAGE2,
        num_warps=NUM_WARPS_STAGE2,
        num_stages=NUM_STAGES_STAGE2,
    )

    return out