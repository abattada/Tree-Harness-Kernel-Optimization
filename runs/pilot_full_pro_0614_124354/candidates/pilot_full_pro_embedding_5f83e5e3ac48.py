import torch
import triton
import triton.language as tl


@triton.jit
def embedding_kernel(
    out_ptr,
    weight_ptr,
    idx_ptr,
    embed_dim: tl.constexpr,
    n_rows: int,
    stride_w: int,
    stride_o: int,
    BLOCK_SIZE: tl.constexpr,
    ITEMS_PER_THREAD: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)

    # grid-stride loop: each program handles a strided subset of output rows
    for row_idx in range(pid, n_rows, grid_size):
        # load the index into the weight table
        idx_val = tl.load(idx_ptr + row_idx)
        # pointer to the start of the chosen weight row
        weight_row_ptr = weight_ptr + idx_val * stride_w
        # pointer to the start of the current output row
        out_row_ptr = out_ptr + row_idx * stride_o

        # construct a 2-D offset tile covering the full embedding dimension:
        # [BLOCK_SIZE, ITEMS_PER_THREAD]  →  BLOCK_SIZE threads * ITEMS_PER_THREAD floats
        offs = (tl.arange(0, BLOCK_SIZE)[:, None] * ITEMS_PER_THREAD +
                tl.arange(0, ITEMS_PER_THREAD)[None, :])

        # load the whole row with an eviction hint: the weight table is large and
        # randomly accessed, so we stream the rows without polluting the L2 cache.
        data = tl.load(
            weight_row_ptr + offs,
            cache_modifier=".cg",          # stream (evict-first) for Ampere+
            eviction_policy="evict_first",
        )
        tl.store(out_row_ptr + offs, data)


def triton_run(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    # weight: (131072, 1024) f32, contiguous; idx: (1048576,) i64
    n_rows = idx.numel()
    embed_dim = weight.shape[1]  # 1024

    # Embedding dimension is a compile-time constant to enable full vectorisation
    # and mask elimination. Choose a decomposition that uses a moderate number of
    # threads (good occupancy) and a reasonable number of floats per thread.
    BLOCK_SIZE: tl.constexpr = 256
    ITEMS_PER_THREAD: tl.constexpr = 4
    assert embed_dim == BLOCK_SIZE * ITEMS_PER_THREAD, (
        f"embed_dim {embed_dim} must equal BLOCK_SIZE * ITEMS_PER_THREAD"
    )

    out = torch.empty((n_rows, embed_dim), dtype=weight.dtype, device=weight.device)

    # Launch a relatively small grid and use a grid-stride loop inside each
    # program. This amortises kernel-launch overhead without sacrificing
    # occupancy. 65536 is large enough to fully occupy all SMs on Blackwell.
    grid_size = min(n_rows, 65536)

    # 256 threads → 8 warps (a well-balanced choice for this workload)
    num_warps = 8

    embedding_kernel[(grid_size,)](
        out, weight, idx,
        embed_dim, n_rows,
        weight.stride(0), out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        ITEMS_PER_THREAD=ITEMS_PER_THREAD,
        num_warps=num_warps,
    )
    return out