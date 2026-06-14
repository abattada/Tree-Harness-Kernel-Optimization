import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    x_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
    GRID_SIZE: tl.constexpr,
):
    """
    Persistent RMS normalization kernel:
      out = x * rsqrt(mean(x^2, dim=-1) + eps)

    Each program loops over multiple rows (grid-stride loop) to amortize
    launch overhead and keep the SMs busy.
    """
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)  # contiguous full-row offsets

    # Grid-stride loop over rows
    for row in range(pid, M, GRID_SIZE):
        x_row_ptr = x_ptr + row * N
        out_row_ptr = out_ptr + row * N

        # Load entire row – stream in, hint to evict early
        x = tl.load(x_row_ptr + offsets, mask=None, eviction_policy="evict_first")

        # Single‑pass RMS
        x_sq = x * x
        sum_sq = tl.sum(x_sq, axis=0)
        mean_sq = sum_sq / N
        rstd = tl.rsqrt(mean_sq + eps)

        out = x * rstd

        # Store result – mark as evict_last if later kernels reuse output
        tl.store(out_row_ptr + offsets, out, mask=None, eviction_policy="evict_last")


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    RMS normalization along the last dimension for inputs of shape [8192, 4096].
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    M, N = x.shape
    assert M == 8192 and N == 4096, "This kernel is specialized for [8192, 4096]"

    out = torch.empty_like(x)

    # One full row per tile, no mask required
    BLOCK = N
    # Persistent grid: one program per SM (or a small multiple)
    # RTX 5090 likely has 128 SMs; a grid size of 128 works well.
    GRID_SIZE = 128

    grid = (GRID_SIZE,)

    rms_norm_kernel[grid](
        x,
        out,
        M=M,
        N=N,
        eps=1e-5,
        BLOCK=BLOCK,
        GRID_SIZE=GRID_SIZE,
        num_warps=8,
        num_stages=1,
    )

    return out