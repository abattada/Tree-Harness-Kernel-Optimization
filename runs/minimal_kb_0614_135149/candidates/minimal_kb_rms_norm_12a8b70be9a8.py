import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    x_ptr,
    out_ptr,
    M,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Persistent RMS normalization kernel.
    Each program processes a subset of rows in a grid‑stride loop,
    reducing launch overhead while keeping full‑row vectorization.
    """
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)

    # Grid‑stride loop over rows
    for row in range(pid, M, num_pids):
        x_row_ptr = x_ptr + row * N
        out_row_ptr = out_ptr + row * N

        offsets = tl.arange(0, BLOCK)

        # Load entire row – streaming data, evict early after use
        x = tl.load(x_row_ptr + offsets, mask=None, eviction_policy="evict_first")

        # Single‑pass RMS: x * rsqrt(mean(x^2, dim=-1) + eps)
        x_sq = x * x
        sum_sq = tl.sum(x_sq, axis=0)          # reduce over the row
        mean_sq = sum_sq / N
        rstd = tl.rsqrt(mean_sq + eps)

        out = x * rstd

        # Store result – evict last if later kernels will reuse it
        tl.store(out_row_ptr + offsets, out, mask=None, eviction_policy="evict_last")


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    Apply RMS normalization along the last dimension (f32[8192, 4096]).
    Expects contiguous input; returns a tensor of the same shape.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    M, N = x.shape

    out = torch.empty_like(x)

    # One tile covers the entire row (N = 4096)
    BLOCK = N

    # Launch a persistent grid where each program loops over multiple rows
    # 1024 programs give a good balance between occupancy and launch overhead
    NUM_PROGRAMS = 1024
    grid = (NUM_PROGRAMS,)

    rms_norm_kernel[grid](
        x, out,
        M,
        N=N,
        eps=1e-5,
        BLOCK=BLOCK,
        num_warps=8,   # solid occupancy / register trade‑off
        num_stages=1,  # no pipelining needed
    )

    return out