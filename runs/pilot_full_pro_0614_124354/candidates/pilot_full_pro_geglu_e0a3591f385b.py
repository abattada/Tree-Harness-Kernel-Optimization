import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def geglu_kernel(
    x_ptr,
    out_ptr,
    M,
    N_half,
    stride_xm,
    stride_xn,
    stride_outm,
    stride_outn,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    GEGLU: split input x into (a, b) along the last dimension, apply
    GELU tanh-approx to a, then multiply by b.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = offs_n < N_half

    # Base pointer to the current row
    row_x_base = x_ptr + pid_m * stride_xm

    # Pointers to a and b columns (a = first half, b = second half)
    a_ptrs = row_x_base + offs_n
    b_ptrs = row_x_base + N_half + offs_n

    a = tl.load(a_ptrs, mask=mask, other=0.0)
    b = tl.load(b_ptrs, mask=mask, other=0.0)

    # GELU tanh approximation:
    # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654  # math.sqrt(2 / math.pi)
    a3 = a * a * a
    inner = a + 0.044715 * a3
    tanh_inp = sqrt_2_over_pi * inner
    gelu = 0.5 * a * (1.0 + libdevice.tanh(tanh_inp))
    result = gelu * b

    # Store output
    out_ptrs = out_ptr + pid_m * stride_outm + offs_n
    tl.store(out_ptrs, result, mask=mask)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    Implements GEGLU: a, b = x.chunk(2, dim=-1); return F.gelu(a, approximate="tanh") * b.
    x: shape [8192, 8192]
    returns: shape [8192, 4096]
    """
    M, N2 = x.shape
    N_half = N2 // 2
    out = torch.empty((M, N_half), device=x.device, dtype=x.dtype)

    # Tunable launch parameters – chosen as sensible defaults.
    BLOCK_SIZE_N = 256   # columns per program
    num_warps = 8        # one warp per 32 elements
    grid = (M, triton.cdiv(N_half, BLOCK_SIZE_N))

    geglu_kernel[grid](
        x,
        out,
        M,
        N_half,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        num_warps=num_warps,
    )
    return out