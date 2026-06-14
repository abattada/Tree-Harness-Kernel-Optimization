import torch
import triton
import triton.language as tl

@triton.jit
def rope_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    stride_x_b: tl.constexpr, stride_x_h: tl.constexpr, stride_x_s: tl.constexpr, stride_x_d: tl.constexpr,
    stride_cos_s: tl.constexpr, stride_cos_d: tl.constexpr,
    stride_sin_s: tl.constexpr, stride_sin_d: tl.constexpr,
    stride_out_b: tl.constexpr, stride_out_h: tl.constexpr, stride_out_s: tl.constexpr, stride_out_d: tl.constexpr,
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    HALF_D: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    start_row = pid * ROWS_PER_PROG
    offs_half = tl.arange(0, HALF_D)
    # Loop over assigned rows, break early if past total_rows
    for i in range(ROWS_PER_PROG):
        row = start_row + i
        # Compute (b, h, s) from linear row index
        b = row // (H * S)
        rem = row % (H * S)
        h = rem // S
        s = rem % S

        # Base pointers (scalar)
        x_base = x_ptr + b * stride_x_b + h * stride_x_h + s * stride_x_s
        out_base = out_ptr + b * stride_out_b + h * stride_out_h + s * stride_out_s
        cos_base = cos_ptr + s * stride_cos_s
        sin_base = sin_ptr + s * stride_sin_s

        # Load x halves with contiguous hints
        x1_ptr = tl.max_contiguous(x_base + offs_half, HALF_D)
        x1 = tl.load(x1_ptr, eviction_policy='evict_first')
        x2_ptr = tl.max_contiguous(x_base + (offs_half + HALF_D), HALF_D)
        x2 = tl.load(x2_ptr, eviction_policy='evict_first')

        # Load cos and sin for this sequence position
        c_ptr = tl.max_contiguous(cos_base + offs_half, HALF_D)
        c = tl.load(c_ptr, eviction_policy='evict_first')
        s_ptr = tl.max_contiguous(sin_base + offs_half, HALF_D)
        s_val = tl.load(s_ptr, eviction_policy='evict_first')

        # Compute rotated halves
        out1 = x1 * c - x2 * s_val
        out2 = x1 * s_val + x2 * c

        # Store output with contiguous hints
        out1_ptr = tl.max_contiguous(out_base + offs_half, HALF_D)
        tl.store(out1_ptr, out1)
        out2_ptr = tl.max_contiguous(out_base + (offs_half + HALF_D), HALF_D)
        tl.store(out2_ptr, out2)


def triton_run(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float16
    assert cos.dtype == torch.float16 and sin.dtype == torch.float16
    B, H, S, D = x.shape
    assert D == 128
    assert cos.shape == (S, D // 2) and sin.shape == (S, D // 2)

    out = torch.empty_like(x)

    HALF_D = D // 2  # 64
    # Increase ROWS_PER_PROG to reduce grid size and launch overhead
    ROWS_PER_PROG = 16
    total_rows = B * H * S
    assert total_rows % ROWS_PER_PROG == 0, "total_rows must be divisible by ROWS_PER_PROG"

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