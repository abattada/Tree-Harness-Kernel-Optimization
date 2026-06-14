import torch
import triton
import triton.language as tl


@triton.jit
def _embedding_kernel(
    weight_ptr,          # [V, D] f32
    idx_ptr,             # [N] i64
    out_ptr,             # [N, D] f32
    D: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Each program loads one index, then loads the corresponding row from weight
    and writes it entirely to out.  The row load/store is vectorized: each thread
    handles a contiguous chunk of (BLOCK_SIZE / num_threads) elements.
    """
    pid = tl.program_id(0)
    if pid < N:
        idx_val = tl.load(idx_ptr + pid)
        idx_val32 = idx_val.to(tl.int32)

        src_base = weight_ptr + idx_val32 * D
        dst_base = out_ptr + pid * D

        offs = tl.arange(0, BLOCK_SIZE)   # BLOCK_SIZE == D, so full row
        val = tl.load(src_base + offs)
        tl.store(dst_base + offs, val)


def triton_run(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Embedding lookup: out[i, :] = weight[idx[i], :]
    weight: (V, D) float32
    idx:    (N,)   int64
    returns: (N, D) float32
    """
    V, D = weight.shape
    N = idx.shape[0]

    # Full row fits in one block – single vectorized load/store.
    BLOCK_SIZE = D

    # Choose a number of warps so that each thread handles a power-of-two
    # number of consecutive elements (e.g. 4 warps → 128 threads → 8 elements/thread).
    NUM_WARPS = 4
    assert D % (NUM_WARPS * 32) == 0, (
        f"D ({D}) must be a multiple of num_warps*32 ({NUM_WARPS*32})"
    )

    out = torch.empty(N, D, dtype=weight.dtype, device=weight.device)
    grid = (N,)
    _embedding_kernel[grid](
        weight, idx, out,
        D, N, BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )
    return out