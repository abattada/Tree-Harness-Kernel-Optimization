import torch
import triton
import triton.language as tl


@triton.jit
def embedding_kernel(
    weight_ptr,         # fp32 [V, D]
    idx_ptr,            # i64  [N]
    out_ptr,            # fp32 [N, D]
    N,                  # number of output rows
    D,                  # embedding dimension
    stride_weight_0,    # leading dim of weight (D if contiguous)
    stride_weight_1,    # inner dim (1 if contiguous)
    stride_out_0,
    stride_out_1,
    BLOCK_D: tl.constexpr,    # compile-time tile size (= D)
    GRID_SIZE: tl.constexpr,  # number of programs (persistent)
):
    pid = tl.program_id(0)
    # vector of offsets covering the whole embedding dimension
    offsets = tl.arange(0, BLOCK_D)  # BLOCK_D == D

    # grid-stride loop: each program processes rows pid, pid+GRID_SIZE, ...
    for row in range(pid, N, GRID_SIZE):
        idx = tl.load(idx_ptr + row)
        weight_row_base = weight_ptr + idx * stride_weight_0

        # load one full embedding row; D divides the tile exactly → no mask
        w = tl.load(
            weight_row_base + offsets * stride_weight_1,
            eviction_policy="evict_first",
        )

        out_row_base = out_ptr + row * stride_out_0
        tl.store(
            out_row_base + offsets * stride_out_1,
            w,
            eviction_policy="evict_first",
        )


def triton_run(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather rows: out[i] = weight[idx[i]]"""
    assert weight.dtype == torch.float32, "weight must be float32"
    assert idx.dtype == torch.int64, "idx must be int64"
    assert weight.is_contiguous(), "weight must be contiguous"
    assert idx.is_contiguous(), "idx must be contiguous"
    V, D = weight.shape
    N = idx.shape[0]
    assert D == 1024, f"Expected D == 1024, got {D}"

    BLOCK_D = D                     # 1024
    GRID_SIZE = 8192                # enough to keep all SMs busy

    out = torch.empty(N, D, dtype=torch.float32, device=weight.device)
    grid = (GRID_SIZE,)

    embedding_kernel[grid](
        weight, idx, out,
        N, D,
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_D=BLOCK_D,
        GRID_SIZE=GRID_SIZE,
        num_warps=4,
        num_stages=2,
    )

    return out