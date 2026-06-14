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
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    # base offsets for this (b, h, s) row
    base_x = pid_b * stride_x_b + pid_h * stride_x_h + pid_s * stride_x_s
    base_out = pid_b * stride_out_b + pid_h * stride_out_h + pid_s * stride_out_s

    offs = tl.arange(0, BLOCK_HALF)  # 0..63

    # load first half x1
    x1 = tl.load(x_ptr + base_x + offs * stride_x_d, mask=offs < HALF_D)
    # load second half x2
    x2 = tl.load(x_ptr + base_x + (offs + HALF_D) * stride_x_d, mask=offs < HALF_D)

    # load cos and sin for this sequence position
    cos_vals = tl.load(cos_ptr + pid_s * stride_cos_s + offs * stride_cos_d, mask=offs < HALF_D)
    sin_vals = tl.load(sin_ptr + pid_s * stride_sin_s + offs * stride_sin_d, mask=offs < HALF_D)

    # compute rotated halves
    out1 = x1 * cos_vals - x2 * sin_vals
    out2 = x1 * sin_vals + x2 * cos_vals

    # store first half
    tl.store(out_ptr + base_out + offs * stride_out_d, out1, mask=offs < HALF_D)
    # store second half
    tl.store(out_ptr + base_out + (offs + HALF_D) * stride_out_d, out2, mask=offs < HALF_D)


def triton_run(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float16
    assert cos.dtype == torch.float16 and sin.dtype == torch.float16
    B, H, S, D = x.shape
    assert D == 128
    assert cos.shape == (S, D // 2) and sin.shape == (S, D // 2)

    out = torch.empty_like(x)

    HALF_D = D // 2          # 64
    BLOCK_HALF = 64          # load entire half at once

    grid = (B, H, S)

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