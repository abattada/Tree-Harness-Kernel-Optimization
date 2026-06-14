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

    # Each program handles a strided subset of rows (grid-stride loop).
    # A small grid (e.g. 4096) lets each program process many rows,
    # amortising launch and scheduling overhead.
    for row_idx in range(pid, n_rows, grid_size):
        idx_val = tl.load(idx_ptr + row_idx)
        weight_row_ptr = weight_ptr + idx_val * stride_w
        out_row_ptr = out_ptr + row_idx * stride_o

        # Construct 2D offset to load the entire embedding row in one vectorised access.
        offs = (
            tl.arange(0, BLOCK_SIZE)[:, None] * ITEMS_PER_THREAD
            + tl.arange(0, ITEMS_PER_THREAD)[None, :]
        )
        data = tl.load(weight_row_ptr + offs, eviction_policy="evict_first")
        tl.store(out_row_ptr + offs, data, eviction_policy="evict_first")


def triton_run(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    # weight: (131072, 1024) f32, contiguous
    # idx:   (1048576,) i64
    n_rows = idx.numel()
    embed_dim = weight.shape[1]  # 1024

    # Decompose embed_dim into block dimensions
    BLOCK_SIZE: int = 256
    ITEMS_PER_THREAD: int = 4
    assert embed_dim == BLOCK_SIZE * ITEMS_PER_THREAD, "embed_dim must equal block * items"

    out = torch.empty((n_rows, embed_dim), dtype=weight.dtype, device=weight.device)

    # Reduced grid size: fewer programs, each looping over many rows.
    # 4096 blocks keep SMs well-fed while lowering launch overhead further.
    grid_size = min(n_rows, 4096)

    num_warps = BLOCK_SIZE // 32  # 8 warps per block
    embedding_kernel[(grid_size,)](
        out,
        weight,
        idx,
        embed_dim,
        n_rows,
        weight.stride(0),
        out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        ITEMS_PER_THREAD=ITEMS_PER_THREAD,
        num_warps=num_warps,
    )
    return out