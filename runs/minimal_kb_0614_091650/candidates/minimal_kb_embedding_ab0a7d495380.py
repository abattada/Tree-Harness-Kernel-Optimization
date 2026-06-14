import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel: gather rows from a weight matrix according to indices.
# Specialized for the known problem size: weight f32[131072, 1024],
# indices i64[1048576].  All dimensions are passed as tl.constexpr so that
# the compiler can eliminate bounds checks and generate fully vectorized code.
# ---------------------------------------------------------------------------
@triton.jit
def embedding_kernel(
    weight_ptr, idx_ptr, out_ptr,
    stride_weight_0, stride_weight_1,
    stride_out_0, stride_out_1,
    D: tl.constexpr,              # embedding dimension, 1024
    N: tl.constexpr,              # number of indices, 1048576
    BLOCK_D: tl.constexpr,        # = D, always a multiple of D
):
    pid = tl.program_id(0)        # flat row index, 0 .. N-1

    # Load the index for this output row (pid)
    idx = tl.load(idx_ptr + pid)

    # Compute the starting address of the selected weight row
    weight_base = weight_ptr + idx * stride_weight_0

    # Vectorized load of the entire weight row (1024 contiguous floats)
    offs = tl.arange(0, BLOCK_D)  # 0 .. 1023
    # No mask needed because BLOCK_D == D and all indices are valid
    w = tl.load(
        weight_base + offs * stride_weight_1,
        eviction_policy='evict_first',
    )

    # Store the result into the output row
    out_base = out_ptr + pid * stride_out_0
    tl.store(
        out_base + offs * stride_out_1,
        w,
        eviction_policy='evict_first',
    )


def triton_run(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    # Assert exactly the shapes and types required by the problem
    assert weight.shape == (131072, 1024), f"Unexpected weight shape {weight.shape}"
    assert idx.shape == (1048576,), f"Unexpected idx shape {idx.shape}"
    assert weight.dtype == torch.float32
    assert idx.dtype == torch.int64
    assert weight.is_contiguous()

    D = weight.shape[1]           # 1024
    N = idx.shape[0]             # 1048576

    out = torch.empty(N, D, dtype=torch.float32, device=weight.device)

    # Launch one program per output row; dimensions are compile-time constants
    grid = (N,)
    embedding_kernel[grid](
        weight, idx, out,
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        D, N, D,              # D, N, BLOCK_D all constexpr (D=1024, N=1048576)
        num_warps=8,          # increased from 4 to improve occupancy
        num_stages=2,
    )

    return out