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
    row = tl.program_id(0)
    x_row_ptr = x_ptr + row * N
    out_row_ptr = out_ptr + row * N

    offsets = tl.arange(0, BLOCK)

    # Give the compiler strong alignment / contiguity hints so it can generate
    # wide vectorized loads and stores.
    offsets = tl.multiple_of(offsets, 16)
    x_row_ptr = tl.multiple_of(x_row_ptr, 16)
    out_row_ptr = tl.multiple_of(out_row_ptr, 16)

    # Load the whole row with an eviction hint – data is streamed, no reuse.
    x = tl.load(x_row_ptr + offsets, mask=None, eviction_policy="evict_first")

    # Single‑pass RMS norm: x * rsqrt(mean(x^2) + eps)
    x_sq = x * x
    sum_sq = tl.sum(x_sq, axis=0)
    mean_sq = sum_sq / N
    rstd = tl.rsqrt(mean_sq + eps)

    out = x * rstd

    # Store the result with a hint that it may be reused by a later consumer.
    tl.store(out_row_ptr + offsets, out, mask=None, eviction_policy="evict_last")


def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous()
    M, N = x.shape
    out = torch.empty_like(x)

    BLOCK = N  # 4096 – one tile equals one full row, no masking needed
    grid = (M,)

    rms_norm_kernel[grid](
        x,
        out,
        N=N,
        eps=1e-5,       # must match the PyTorch reference
        BLOCK=BLOCK,
        num_warps=4,    # 128 threads per block: better occupancy on Blackwell
        num_stages=1,   # no pipelining needed for a single load/store
    )
    return out