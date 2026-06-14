import torch
import triton
import triton.language as tl


@triton.jit
def _embedding_kernel(weight_ptr, idx_ptr, out_ptr,
                      D: tl.constexpr, N: tl.constexpr,
                      BLOCK_SIZE: tl.constexpr):
    """
    Each program handles one lookup row.
    BLOCK_SIZE must equal D so that the entire embedding vector is loaded in
    one vectorized, mask‑free instruction.
    """
    row_id = tl.program_id(0)
    if row_id < N:
        idx_val = tl.load(idx_ptr + row_id)
        idx_val32 = tl.cast(idx_val, tl.int32)

        # All offsets for the embedding dimension
        offs = tl.arange(0, BLOCK_SIZE)                 # = D
        # Base address of the selected row
        row_base = weight_ptr + idx_val32 * D

        # Since BLOCK_SIZE == D and D is a multiple of the vector width,
        # there is no need for a mask – the load is always in‑bounds.
        vals = tl.load(row_base + offs)
        tl.store(out_ptr + row_id * D + offs, vals)


def triton_run(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Embedding lookup: out[i, :] = weight[idx[i], :]
    weight: (V, D) float32
    idx:    (N,)   int64
    returns: (N, D) float32
    """
    V, D = weight.shape
    N = idx.shape[0]

    # Set the block size to the full embedding dimension so that every load
    # is a wide, coalesced, single‑instruction access.  Each thread processes
    # D / (num_warps * 32) elements.
    BLOCK_SIZE = D
    # 16 warps → 512 threads, each handles 1024/512 = 2 float32 elements.
    # This balances occupancy and instruction‑level vectorisation.
    num_warps = 16
    assert D % (num_warps * 32) == 0, f"D={D} must be divisible by {num_warps*32}"

    out = torch.empty(N, D, dtype=weight.dtype, device=weight.device)
    grid = (N,)
    _embedding_kernel[grid](
        weight, idx, out, D, N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return out