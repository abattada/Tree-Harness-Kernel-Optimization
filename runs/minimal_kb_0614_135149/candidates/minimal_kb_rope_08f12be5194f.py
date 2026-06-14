import torch
import triton
import triton.language as tl


@triton.jit
def rope_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    stride_x_b,
    stride_x_h,
    stride_x_s,
    stride_x_d,
    stride_cos_s,
    stride_cos_d,
    stride_sin_s,
    stride_sin_d,
    stride_out_b,
    stride_out_h,
    stride_out_s,
    stride_out_d,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    HALF_D: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    """Optimized RoPE kernel with increased rows per program and higher occupancy."""
    pid = tl.program_id(0)
    start_row = pid * ROWS_PER_PROG
    offs_half = tl.arange(0, HALF_D)  # 0..63

    for i in range(ROWS_PER_PROG):
        row = start_row + i
        b = row // (H * S)
        rem = row % (H * S)
        h = rem // S
        s = rem % S

        # Base pointers for the current row
        x_base = x_ptr + b * stride_x_b + h * stride_x_h + s * stride_x_s
        out_base = out_ptr + b * stride_out_b + h * stride_out_h + s * stride_out_s

        # Load the two halves of the embedding (contiguous along last dim)
        x1 = tl.load(x_base + offs_half, eviction_policy='evict_first')
        x2 = tl.load(x_base + offs_half + HALF_D, eviction_policy='evict_first')

        # Load cos/sin for the current sequence position
        c = tl.load(cos_ptr + s * stride_cos_s + offs_half, eviction_policy='evict_first')
        s_vals = tl.load(sin_ptr + s * stride_sin_s + offs_half, eviction_policy='evict_first')

        # Apply rotation: half1' = x1*c - x2*s, half2' = x1*s + x2*c
        out1 = x1 * c - x2 * s_vals
        out2 = x1 * s_vals + x2 * c

        # Store the transformed halves
        tl.store(out_base + offs_half, out1)
        tl.store(out_base + offs_half + HALF_D, out2)


def triton_run(x, cos, sin):
    """Run RoPE: rotate half the last dimension of x using cos/sin tables."""
    assert x.dtype == torch.float16
    assert cos.dtype == torch.float16 and sin.dtype == torch.float16
    B, H, S, D = x.shape
    assert D == 128, "expected last dim 128"
    assert cos.shape == (S, D // 2) and sin.shape == (S, D // 2), "cos/sin shape mismatch"

    out = torch.empty_like(x)

    HALF_D = D // 2  # 64
    ROWS_PER_PROG = 16  # increased from 8 to process more rows per block
    total_rows = B * H * S
    assert total_rows % ROWS_PER_PROG == 0, "total rows must divide ROWS_PER_PROG exactly"

    grid = (total_rows // ROWS_PER_PROG,)

    rope_kernel[grid](
        x, cos, sin, out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B=B, H=H, S=S, D=D,
        HALF_D=HALF_D,
        ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=8,     # increased from 4 to boost SM occupancy
        num_stages=2,    # kept at 2 for minimal shared memory contention
    )
    return out