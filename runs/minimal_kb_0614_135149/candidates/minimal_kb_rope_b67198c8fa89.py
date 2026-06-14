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
    BH: int,                     # runtime total = B*H, keeps loop dynamic
):
    """
    Each program handles a single sequence position s, reusing the cos/sin
    loads across all batch*head rows for that s.
    """
    s = tl.program_id(0)
    offs_half = tl.arange(0, HALF_D)

    # Load cos/sin for this s once – they will be kept in registers.
    c = tl.load(cos_ptr + s * stride_cos_s + offs_half * stride_cos_d)
    s_vals = tl.load(sin_ptr + s * stride_sin_s + offs_half * stride_sin_d)

    # Iterate over all (b, h) combinations dynamically (not unrolled).
    for idx in range(BH):
        b = idx // H
        h = idx % H

        # Base pointers for this (b, h, s) row.
        base_x = b * stride_x_b + h * stride_x_h + s * stride_x_s
        base_out = b * stride_out_b + h * stride_out_h + s * stride_out_s

        # Load the two halves of x (contiguous, 64 elements each).
        x1 = tl.load(base_x + offs_half * stride_x_d, eviction_policy='evict_first')
        x2 = tl.load(base_x + (offs_half + HALF_D) * stride_x_d, eviction_policy='evict_first')

        # RoPE arithmetic.
        out1 = x1 * c - x2 * s_vals
        out2 = x1 * s_vals + x2 * c

        # Store rotated halves.
        tl.store(base_out + offs_half * stride_out_d, out1)
        tl.store(base_out + (offs_half + HALF_D) * stride_out_d, out2)


def triton_run(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float16
    assert cos.dtype == torch.float16 and sin.dtype == torch.float16
    B, H, S, D = x.shape
    assert D == 128, "expected last dim 128"
    assert cos.shape == (S, D // 2) and sin.shape == (S, D // 2), "cos/sin shape mismatch"

    out = torch.empty_like(x)
    HALF_D = D // 2           # 64
    BH: int = B * H          # 8 * 32 = 256

    # One program per sequence position -> grid = S.
    grid = (S,)

    rope_kernel[grid](
        x, cos, sin, out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, S, D,
        HALF_D,
        BH,
        num_warps=2,          # 64 threads exactly cover the 64-element half
        num_stages=2,
    )
    return out