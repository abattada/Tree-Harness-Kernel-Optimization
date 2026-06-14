import torch
import triton
import triton.language as tl

@triton.jit
def rope_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    stride_x_b, stride_x_h, stride_x_s, stride_x_d,
    stride_out_b, stride_out_h, stride_out_s, stride_out_d,
    stride_cos_s, stride_cos_half,
    stride_sin_s, stride_sin_half,
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    HALF: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    # Offset for x row
    x_offset = pid_b * stride_x_b + pid_h * stride_x_h + pid_s * stride_x_s
    x_ptr_row = x_ptr + x_offset

    # Offset for output row
    out_offset = pid_b * stride_out_b + pid_h * stride_out_h + pid_s * stride_out_s
    out_ptr_row = out_ptr + out_offset

    # Offset for cos/sin row (only need the s dimension)
    cos_row_ptr = cos_ptr + pid_s * stride_cos_s
    sin_row_ptr = sin_ptr + pid_s * stride_sin_s

    # Load x row (128 elements)
    offsets_d = tl.arange(0, BLOCK)
    x = tl.load(x_ptr_row + offsets_d * stride_x_d, mask=offsets_d < D, other=0.0).to(tl.float32)

    # Load cos and sin (64 elements each)
    offsets_half = tl.arange(0, HALF)
    cos_vals = tl.load(cos_row_ptr + offsets_half * stride_cos_half, mask=offsets_half < HALF, other=0.0).to(tl.float32)
    sin_vals = tl.load(sin_row_ptr + offsets_half * stride_sin_half, mask=offsets_half < HALF, other=0.0).to(tl.float32)

    # Split x into x1 (first half) and x2 (second half)
    x1 = tl.where(offsets_d < HALF, x, 0.0)  # shape (BLOCK,) but only first 64 nonzero
    x2 = tl.where(offsets_d >= HALF, x, 0.0)  # second half

    # Compute rotation
    # For first half indices: out = x1 * cos - x2 * sin
    out_first = x1 * cos_vals - x2 * sin_vals
    # For second half indices: out = x1 * sin + x2 * cos
    out_second = x1 * sin_vals + x2 * cos_vals

    # Combine using conditional, but we have already masked x1/x2 so we can sum
    out = out_first + out_second

    # Store output
    tl.store(out_ptr_row + offsets_d * stride_out_d, out.to(tl.float16), mask=offsets_d < D)


def triton_run(x, cos, sin):
    assert x.dtype == torch.float16
    assert cos.dtype == torch.float16
    assert sin.dtype == torch.float16
    B, H, S, D = x.shape
    assert D == 128
    assert cos.shape == (S, D//2)
    assert sin.shape == (S, D//2)

    out = torch.empty_like(x)

    BLOCK = 128
    HALF = 64

    grid = (B, H, S)

    rope_kernel[grid](
        x, cos, sin, out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        B, H, S, D,
        HALF, BLOCK,
        num_warps=4,
        num_stages=3
    )
    return out