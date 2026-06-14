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

    # each program processes a strided subset of the output rows
    for row_idx in range(pid, n_rows, grid_size):
        # load the index into weight
        idx_val = tl.load(idx_ptr + row_idx)
        # pointer to the start of the chosen weight row
        weight_row_ptr = weight_ptr + idx_val * stride_w
        # pointer to the current output row
        out_row_ptr = out_ptr + row_idx * stride_o

        # construct 2D offset block: [BLOCK_SIZE, ITEMS_PER_THREAD]
        offs = (tl.arange(0, BLOCK_SIZE)[:, None] * ITEMS_PER_THREAD +
                tl.arange(0, ITEMS_PER_THREAD)[None, :])
        data = tl.load(weight_row_ptr + offs)
        tl.store(out_row_ptr + offs, data)


def triton_run(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    # weight: (131072, 1024) f32, contiguous; idx: (1048576,) i64
    n_rows = idx.numel()
    embed_dim = weight.shape[1]  # 1024

    # tuneable block decomposition: embed_dim = BLOCK_SIZE * ITEMS_PER_THREAD
    BLOCK_SIZE = 256
    ITEMS_PER_THREAD = 4
    assert embed_dim == BLOCK_SIZE * ITEMS_PER_THREAD, "embed_dim must equal block * items"

    # allocate output
    out = torch.empty((n_rows, embed_dim), dtype=weight.dtype, device=weight.device)

    # grid size: a smaller grid + grid-stride loop amortises launch overhead
    grid_size = min(n_rows, 65536)

    num_warps = BLOCK_SIZE // 32  # 8 warps
    embedding_kernel[(grid_size,) ](
        out, weight, idx,
        embed_dim, n_rows,
        weight.stride(0), out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        ITEMS_PER_THREAD=ITEMS_PER_THREAD,
        num_warps=num_warps,
    )
    return out