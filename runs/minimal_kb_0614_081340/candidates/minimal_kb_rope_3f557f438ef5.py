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
    BLOCK: tl.constexpr,
):
    # Program IDs for batch, head, sequence
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    # Base pointers for this (b, h, s)
    x_base = x_ptr + pid_b * stride_x_b + pid_h * stride_x_h + pid_s * stride_x_s
    out_base = out_ptr + pid_b * stride_out_b + pid_h * stride_out_h + pid_s * stride_out_s
    cos_base = cos_ptr + pid_s * stride_cos_s  # shape (S, D/2)
    sin_base = sin_ptr + pid_s * stride_sin_s

    offs = tl.arange(0, BLOCK)

    # Load x1 (first half) and x2 (second half)
    x1 = tl.load(x_base + offs, offs)                     # x[0:64]
    x2 = tl.load(x_base + offs + BLOCK, offs + BLOCK)    # x[64:128]

    # Load cos and sin (both 64-element vectors)
    cos_vals = tl.load(cos_base + offs, offs)
    sin_vals = tl.load(sin_base + offs, offs)

    # Compute RoPE halves
    out1 = x1 * cos_vals - x2 * sin_vals
    out2 = x1 * sin_vals + x2 * cos_vals

    # Store output halves
    tl.store(out_base + offs, out1, offs)
    tl.store(out_base + offs + BLOCK, out2, offs + BLOCK)


def triton_run(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float16
    assert cos.dtype == torch.float16
    assert sin.dtype == torch.float16
    B, H, S, D = x.shape
    assert D == 128
    assert cos.shape == (S, 64)
    assert sin.shape == (S, 64)

    out = torch.empty_like(x)

    BLOCK = 64  # half of the dimension

    grid = (B, H, S)

    rope_kernel[grid](
        x, cos, sin, out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK,
        num_warps=4,
        num_stages=1,
    )

    return out