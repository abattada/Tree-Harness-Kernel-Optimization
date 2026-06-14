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
    BLOCK_HALF: tl.constexpr,
):
    """
    Flat grid: one program per (b, h, s) row.
    RoPE rotate-half: out = cat(x1*cos - x2*sin, x1*sin + x2*cos)
    """
    pid = tl.program_id(0)
    total_rows = B * H * S
    if pid >= total_rows:
        return

    # decompose pid to (b, h, s)
    b = pid // (H * S)
    rem = pid % (H * S)
    h = rem // S
    s = rem % S

    # base offsets for this row
    base_x = b * stride_x_b + h * stride_x_h + s * stride_x_s
    base_out = b * stride_out_b + h * stride_out_h + s * stride_out_s

    # offsets along the feature dimension (half)
    offs = tl.arange(0, BLOCK_HALF)  # 0..63

    # load first half x1
    x1 = tl.load(x_ptr + base_x + offs * stride_x_d)
    # load second half x2
    x2 = tl.load(x_ptr + base_x + (offs + HALF_D) * stride_x_d)

    # load cos and sin for this sequence position
    cos_vals = tl.load(cos_ptr + s * stride_cos_s + offs * stride_cos_d)
    sin_vals = tl.load(sin_ptr + s * stride_sin_s + offs * stride_sin_d)

    # compute rotated halves
    out1 = x1 * cos_vals - x2 * sin_vals
    out2 = x1 * sin_vals + x2 * cos_vals

    # store output halves
    tl.store(out_ptr + base_out + offs * stride_out_d, out1)
    tl.store(out_ptr + base_out + (offs + HALF_D) * stride_out_d, out2)


def triton_run(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float16
    assert cos.dtype == torch.float16 and sin.dtype == torch.float16

    B, H, S, D = x.shape
    assert D == 128, "RoPE kernel expects D=128"
    assert cos.shape == (S, D // 2) and sin.shape == (S, D // 2)

    out = torch.empty_like(x)

    HALF_D = D // 2          # 64
    BLOCK_HALF = 64          # load entire half at once

    grid = (B * H * S,)      # flat grid: one program per row

    rope_kernel[grid](
        x, cos, sin, out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, S, D,
        HALF_D, BLOCK_HALF,
        num_warps=4,
        num_stages=2,
    )

    return out