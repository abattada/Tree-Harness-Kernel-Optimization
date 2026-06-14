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
    """RoPE kernel: processes ROWS_PER_PROG rows per program with vectorised loads."""
    pid = tl.program_id(0)
    start_row = pid * ROWS_PER_PROG
    offs_half = tl.arange(0, HALF_D)  # 0..63

    for i in range(ROWS_PER_PROG):
        row = start_row + i
        b = row // (H * S)
        rem = row % (H * S)
        h = rem // S
        s = rem % S

        base_x = b * stride_x_b + h * stride_x_h + s * stride_x_s
        base_out = b * stride_out_b + h * stride_out_h + s * stride_out_s

        # Load x halves with contiguous hint for better vectorisation
        x1 = tl.load(
            x_ptr + base_x + offs_half * stride_x_d,
            eviction_policy='evict_first',
            max_contiguous=64,
        )
        x2 = tl.load(
            x_ptr + base_x + (offs_half + HALF_D) * stride_x_d,
            eviction_policy='evict_first',
            max_contiguous=64,
        )

        c = tl.load(
            cos_ptr + s * stride_cos_s + offs_half * stride_cos_d,
            eviction_policy='evict_first',
            max_contiguous=64,
        )
        s_vals = tl.load(
            sin_ptr + s * stride_sin_s + offs_half * stride_sin_d,
            eviction_policy='evict_first',
            max_contiguous=64,
        )

        out1 = x1 * c - x2 * s_vals
        out2 = x1 * s_vals + x2 * c

        tl.store(
            out_ptr + base_out + offs_half * stride_out_d,
            out1,
            max_contiguous=64,
        )
        tl.store(
            out_ptr + base_out + (offs_half + HALF_D) * stride_out_d,
            out2,
            max_contiguous=64,
        )


def triton_run(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Run RoPE: rotate half the last dimension of x using cos/sin tables."""
    assert x.dtype == torch.float16
    assert cos.dtype == torch.float16 and sin.dtype == torch.float16
    B, H, S, D = x.shape
    assert D == 128
    assert cos.shape == (S, D // 2) and sin.shape == (S, D // 2)

    out = torch.empty_like(x)

    HALF_D = D // 2  # 64
    ROWS_PER_PROG = 16   # increased from 8 to reduce launch overhead
    total_rows = B * H * S
    assert total_rows % ROWS_PER_PROG == 0, "ROWS_PER_PROG must divide total rows exactly"

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
        num_warps=2,   # exactly 64 threads, perfect for the 64-element loads
        num_stages=3,  # slight pipelining improvement over 2
    )
    return out