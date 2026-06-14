import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

@triton.jit
def welford_partial_kernel(
    x_ptr, partials_ptr, N,
    BLOCK_SIZE: tl.constexpr, NUM_CHUNKS: tl.constexpr
):
    """
    Each thread block processes a contiguous chunk of one row and computes
    (n, mean, M2) for that chunk using a single-pass Welford reduction.
    Result stored into partials_ptr[row, chunk, :3].
    """
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    tid = tl.local_id(0)

    # Load single element
    offs = row * N + chunk * BLOCK_SIZE + tid
    x_val = tl.load(x_ptr + offs)

    # Per-element Welford state
    n = 1.0
    mean = x_val
    M2 = 0.0

    # Shared memory for block reduction
    counts = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    means = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    M2s = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    counts[tid] = n
    means[tid] = mean
    M2s[tid] = M2
    tl.barrier()

    # Tree reduction merging Welford states
    stride = 1
    while stride < BLOCK_SIZE:
        if tid % (2 * stride) == 0:
            other = tid + stride
            n1 = counts[tid]
            m1 = means[tid]
            m2_1 = M2s[tid]

            n2 = counts[other]
            m2_val = means[other]
            m2_2 = M2s[other]

            n_new = n1 + n2
            delta = m2_val - m1
            new_mean = m1 + delta * n2 / n_new
            new_M2 = m2_1 + m2_2 + delta * delta * n1 * n2 / n_new

            counts[tid] = n_new
            means[tid] = new_mean
            M2s[tid] = new_M2
        tl.barrier()
        stride *= 2

    if tid == 0:
        base = partials_ptr + row * NUM_CHUNKS * 3 + chunk * 3
        tl.store(base + 0, counts[0])
        tl.store(base + 1, means[0])
        tl.store(base + 2, M2s[0])


@triton.jit
def welford_combine_kernel(
    partials_ptr, out_ptr, N, NUM_CHUNKS, M,
    BLOCK_SIZE: tl.constexpr
):
    """
    One thread block per row merges all chunk partials and writes
    final mean and population variance into out_ptr (shape [2, M]).
    """
    row = tl.program_id(0)
    tid = tl.local_id(0)

    # Load the partials for this row; idle threads contribute identity (n=0)
    base = partials_ptr + row * NUM_CHUNKS * 3
    n = tl.where(tid < NUM_CHUNKS, tl.load(base + tid * 3 + 0), 0.0)
    mean = tl.where(tid < NUM_CHUNKS, tl.load(base + tid * 3 + 1), 0.0)
    M2 = tl.where(tid < NUM_CHUNKS, tl.load(base + tid * 3 + 2), 0.0)

    # Shared memory reduction
    counts = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    means = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    M2s = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    counts[tid] = n
    means[tid] = mean
    M2s[tid] = M2
    tl.barrier()

    stride = 1
    while stride < BLOCK_SIZE:
        if tid % (2 * stride) == 0:
            other = tid + stride
            n1 = counts[tid]
            m1 = means[tid]
            m2_1 = M2s[tid]

            n2 = counts[other]
            m2_val = means[other]
            m2_2 = M2s[other]

            n_new = n1 + n2
            delta = m2_val - m1
            new_mean = m1 + delta * n2 / n_new
            new_M2 = m2_1 + m2_2 + delta * delta * n1 * n2 / n_new

            counts[tid] = n_new
            means[tid] = new_mean
            M2s[tid] = new_M2
        tl.barrier()
        stride *= 2

    if tid == 0:
        final_mean = means[0]
        final_M2 = M2s[0]
        var = final_M2 / N   # population variance (N known to be 4096)
        tl.store(out_ptr + row, final_mean)       # out[0, row]
        tl.store(out_ptr + M + row, var)         # out[1, row]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    Compute per-row mean and population variance for a float32 tensor
    of shape (8192, 4096). Returns tensor of shape (2, 8192).
    """
    M, N = x.shape
    # Protect against misuse but stay silent otherwise
    assert x.dtype == torch.float32, "Input must be float32"
    assert x.is_contiguous(), "Input must be contiguous"

    BLOCK_SIZE = 256
    NUM_CHUNKS = N // BLOCK_SIZE          # 16
    COMBINE_BLOCK_SIZE = 32

    partials = torch.empty((M, NUM_CHUNKS, 3), device=x.device, dtype=x.dtype)
    out = torch.empty((2, M), device=x.device, dtype=x.dtype)

    grid_partial = (M, NUM_CHUNKS)
    welford_partial_kernel[grid_partial](
        x, partials, N,
        BLOCK_SIZE=BLOCK_SIZE, NUM_CHUNKS=NUM_CHUNKS,
    )

    grid_combine = (M,)
    welford_combine_kernel[grid_combine](
        partials, out, N, NUM_CHUNKS, M,
        BLOCK_SIZE=COMBINE_BLOCK_SIZE,
    )

    return out