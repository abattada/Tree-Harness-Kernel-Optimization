import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def geglu_kernel(
    x_ptr, y_ptr,
    M, N,
    BM: tl.constexpr, BN: tl.constexpr,
):
    """2D block kernel for geglu: each program processes BM rows x BN columns."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)[:, None]   # shape (BM, 1)
    offs_n = pid_n * BN + tl.arange(0, BN)[None, :]   # shape (1, BN)

    mask = (offs_m < M) & (offs_n < N)

    # Left half and right half indices (input has 2*N columns)
    a_offs = offs_m * (2 * N) + offs_n
    b_offs = offs_m * (2 * N) + offs_n + N

    a = tl.load(x_ptr + a_offs, mask=mask,
                cache_modifier='.cg', eviction_policy='evict_first')
    b = tl.load(x_ptr + b_offs, mask=mask,
                cache_modifier='.cg', eviction_policy='evict_first')

    # GELU with tanh approximation
    sqrt2pi  = 0.7978845608028654
    coeff    = 0.044715
    a3       = a * a * a
    inner    = sqrt2pi * (a + coeff * a3)
    gelu     = 0.5 * a * (1.0 + libdevice.tanh(inner))

    out = gelu * b
    out_offs = offs_m * N + offs_n
    tl.store(y_ptr + out_offs, out, mask=mask, eviction_policy='evict_first')


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """GEGLU: x.shape = (8192, 8192) -> output shape (8192, 4096)."""
    assert x.is_cuda and x.dtype == torch.float32
    M, twoN = x.shape
    N = twoN // 2
    assert twoN == 2 * N, "Input last dimension must be even (a, b chunks)"

    out = torch.empty(M, N, dtype=x.dtype, device=x.device)

    # Tuned block shape: one row per block, all columns.
    # Perfect coalescing, no partial rows, minimized block count.
    BM = 1
    BN = N   # 4096

    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    geglu_kernel[grid](
        x, out, M, N,
        BM=BM, BN=BN,
        num_warps=16, num_stages=4,
    )
    return out