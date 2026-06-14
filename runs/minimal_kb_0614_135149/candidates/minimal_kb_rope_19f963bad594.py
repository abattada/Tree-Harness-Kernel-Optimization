import torch
import triton
import triton.language as tl


@triton.jit
def rope_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    stride_x_b, stride_x_h, stride_x_s, stride_x_d,
    stride_cos_s, stride_cos_d,
    stride_sin_s, stride_sin_d,
    stride_out_b, stride_out_h, stride_out_s, stride_out_d,
    B, H, S, D,
    HALF_D: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    """
    RoPE: rotate half of last dimension.
    Each program processes ROWS_PER_PROG consecutive rows.
    Within a row, we load x1(0:64), x2(64:128), cos[s,:], sin[s,:],
    compute the two output halves, and store.
    Vectorised over the 64-element half-dimension for coalesced access.
    """
    pid = tl.program_id(0)
    start_row = pid * ROWS_PER_PROG

    # Offsets for the half-dimension (0 .. HALF_D-1)
    offs_half = tl.arange(0, HALF_D)

    for i in range(ROWS_PER_PROG):
        row = start_row + i

        # Decompose linear row index into (b, h, s)
        b = row // (H * S)
        rem = row % (H * S)
        h = rem // S
        s = rem % S

        # Base pointers for x and output at this (b,h,s)
        x_base = x_ptr + b * stride_x_b + h * stride_x_h + s * stride_x_s
        out_base = out_ptr + b * stride_out_b + h * stride_out_h + s * stride_out_s

        # Load the two halves of x (contiguous along the inner dimension)
        x1 = tl.load(x_base + offs_half * stride_x_d, eviction_policy='evict_first')
        x2 = tl.load(x_base + (offs_half + HALF_D) * stride_x_d, eviction_policy='evict_first')

        # Load cos and sin for this sequence position (shared across B,H)
        cos_base = cos_ptr + s * stride_cos_s
        sin_base = sin_ptr + s * stride_sin_s
        c = tl.load(cos_base + offs_half * stride_cos_d, eviction_policy='evict_first')
        s_val = tl.load(sin_base + offs_half * stride_sin_d, eviction_policy='evict_first')

        # RoPE computation
        out1 = x1 * c - x2 * s_val
        out2 = x1 * s_val + x2 * c

        # Store output halves
        tl.store(out_base + offs_half * stride_out_d, out1)
        tl.store(out_base + (offs_half + HALF_D) * stride_out_d, out2)


def triton_run(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Applies the RoPE rotation: out = [x1*cos - x2*sin, x1*sin + x2*cos]
    where x1, x2 are the two halves of the last dimension of x.

    x:  (B, H, S, D)   with D = 128
    cos, sin:  (S, D//2)  i.e. (S, 64)
    """
    assert x.dtype == torch.float16, "x must be f16"
    assert cos.dtype == torch.float16 and sin.dtype == torch.float16
    B, H, S, D = x.shape
    HALF_D = D // 2
    assert D == 128
    assert cos.shape == (S, HALF_D) and sin.shape == (S, HALF_D)

    out = torch.empty_like(x)

    # Tuneable parameters; these can be swept for optimal performance.
    ROWS_PER_PROG = 8
    total_rows = B * H * S
    assert total_rows % ROWS_PER_PROG == 0, (
        "total_rows must be divisible by ROWS_PER_PROG; use a grid-stride loop if this isn't guaranteed"
    )
    grid = (total_rows // ROWS_PER_PROG,)

    rope_kernel[grid](
        x, cos, sin, out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, S, D,
        HALF_D,
        ROWS_PER_PROG,
        num_warps=4,
        num_stages=2,
    )

    return out