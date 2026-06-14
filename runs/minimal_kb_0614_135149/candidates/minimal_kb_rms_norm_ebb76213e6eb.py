import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    x_ptr,
    out_ptr,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One program per row; BLOCK == N, so each thread block handles exactly one whole row.
    row = tl.program_id(0)

    # Base pointers for this row
    x_base = x_ptr + row * N
    out_base = out_ptr + row * N

    # Alignment hints to help the compiler generate optimal vectorised loads/stores
    x_base = tl.multiple_of(x_base, 128)
    out_base = tl.multiple_of(out_base, 128)

    # Contiguous offsets covering the whole row, no masking needed
    offsets = tl.arange(0, BLOCK)

    # Load input – streaming access, hint to evict from cache early
    x = tl.load(x_base + offsets, mask=None, eviction_policy="evict_first")

    # Single‑pass RMS norm: x * rsqrt(mean(x^2) + eps)
    x_sq = x * x
    sum_sq = tl.sum(x_sq, axis=0)
    mean_sq = sum_sq / N
    rstd = tl.rsqrt(mean_sq + eps)

    out = x * rstd

    # Store output – may be reused, hint to keep in cache
    tl.store(out_base + offsets, out, mask=None, eviction_policy="evict_last")


def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous()
    M, N = x.shape  # 8192, 4096
    out = torch.empty_like(x)

    BLOCK = N          # one tile per row fits exactly
    grid = (M,)        # one block per row

    rms_norm_kernel[grid](
        x, out,
        N=N,
        eps=1e-5,
        BLOCK=BLOCK,
        num_warps=16,      # increased from 8 to better utilise SM resources
        num_stages=1,      # no software pipelining needed (no loop)
    )
    return out