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
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr,
    HALF_D: tl.constexpr,
):
    # Each program handles one (b, h) pair, looping over all s in 0..S-1
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    # Base pointers for this (b, h)
    x_base = x_ptr + b * stride_x_b + h * stride_x_h
    out_base = out_ptr + b * stride_out_b + h * stride_out_h
    # cos and sin are per s, starting at base
    cos_base = cos_ptr
    sin_base = sin_ptr

    offs_half = tl.arange(0, HALF_D)

    s = 0
    while s < S:
        # Offsets for this sequence position
        x_row = x_base + s * stride_x_s
        out_row = out_base + s * stride_out_s
        cos_cur = cos_base + s * stride_cos_s
        sin_cur = sin_base + s * stride_sin_s

        # Load the two halves of x (contiguous)
        x1 = tl.load(x_row + offs_half * stride_x_d, eviction_policy='evict_first')
        x2 = tl.load(x_row + (offs_half + HALF_D) * stride_x_d, eviction_policy='evict_first')

        # Load cos and sin for this s
        c = tl.load(cos_cur + offs_half * stride_cos_d, eviction_policy='evict_first')
        s_val = tl.load(sin_cur + offs_half * stride_sin_d, eviction_policy='evict_first')

        # Compute rotation
        out1 = x1 * c - x2 * s_val
        out2 = x1 * s_val + x2 * c

        # Store output halves
        tl.store(out_row + offs_half * stride_out_d, out1)
        tl.store(out_row + (offs_half + HALF_D) * stride_out_d, out2)

        s = s + 1


def triton_run(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float16
    assert cos.dtype == torch.float16 and sin.dtype == torch.float16
    B, H, S, D = x.shape
    assert D == 128
    assert cos.shape == (S, 64) and sin.shape == (S, 64)

    out = torch.empty_like(x)

    HALF_D = 64
    grid = (B * H,)  # one program per (batch, head)

    rope_kernel[grid](
        x, cos, sin, out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, S,
        HALF_D,
        num_warps=4,
        num_stages=2,
    )
    return out