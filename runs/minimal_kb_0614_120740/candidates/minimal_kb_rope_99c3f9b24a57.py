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
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    HALF_D: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    # Each program processes ROWS_PER_PROG consecutive rows.
    pid = tl.program_id(0)
    start_row = pid * ROWS_PER_PROG
    offs_half = tl.arange(0, HALF_D)

    for i in range(ROWS_PER_PROG):
        row = start_row + i
        # Decompose linear row index into (b, h, s)
        b = row // (H * S)
        rem = row % (H * S)
        h = rem // S
        s = rem % S

        # Base pointers for x and output at this row
        x_base = x_ptr + b * stride_x_b + h * stride_x_h + s * stride_x_s
        out_base = out_ptr + b * stride_out_b + h * stride_out_h + s * stride_out_s

        # Load x halves (no mask needed because offsets always < D)
        # Use tl.max_contiguous to hint contiguous access
        x1_ptr = x_base + offs_half * stride_x_d
        x1 = tl.load(tl.max_contiguous(x1_ptr, HALF_D), eviction_policy='evict_first')
        x2_ptr = x_base + (offs_half + HALF_D) * stride_x_d
        x2 = tl.load(tl.max_contiguous(x2_ptr, HALF_D), eviction_policy='evict_first')

        # Load cos and sin for this sequence position
        cos_base = cos_ptr + s * stride_cos_s
        sin_base = sin_ptr + s * stride_sin_s
        c_ptr = cos_base + offs_half * stride_cos_d
        c = tl.load(tl.max_contiguous(c_ptr, HALF_D), eviction_policy='evict_first')
        s_ptr = sin_base + offs_half * stride_sin_d
        s_val = tl.load(tl.max_contiguous(s_ptr, HALF_D), eviction_policy='evict_first')

        # Compute rotated halves
        out1 = x1 * c - x2 * s_val
        out2 = x1 * s_val + x2 * c

        # Store output
        tl.store(out_base + offs_half * stride_out_d, out1)
        tl.store(out_base + (offs_half + HALF_D) * stride_out_d, out2)


def triton_run(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float16
    assert cos.dtype == torch.float16 and sin.dtype == torch.float16
    B, H, S, D = x.shape
    assert D == 128
    assert cos.shape == (S, D // 2) and sin.shape == (S, D // 2)

    out = torch.empty_like(x)

    HALF_D = D // 2  # 64
    ROWS_PER_PROG = 8
    total_rows = B * H * S
    assert total_rows % ROWS_PER_PROG == 0, "grid-stride loop assumes exact division"

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
        num_warps=2,      # reduced from 4 to 2 warps for better occupancy
        num_stages=2,
    )
    return out