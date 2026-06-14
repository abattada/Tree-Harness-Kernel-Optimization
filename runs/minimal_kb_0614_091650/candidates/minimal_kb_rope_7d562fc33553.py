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
    HALF_D: tl.constexpr, BLOCK_HALF: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    # Base offsets for x and output at (b, h, s)
    base_x = pid_b * stride_x_b + pid_h * stride_x_h + pid_s * stride_x_s
    base_out = pid_b * stride_out_b + pid_h * stride_out_h + pid_s * stride_out_s

    # Offsets along the last dimension (half-size)
    offs = tl.arange(0, BLOCK_HALF)  # 0..63

    # Load first half x1 (0..63)
    x1_ptrs = x_ptr + base_x + offs * stride_x_d
    x1 = tl.load(x1_ptrs, mask=offs < HALF_D)

    # Load second half x2 (64..127)
    x2_ptrs = x_ptr + base_x + (offs + HALF_D) * stride_x_d
    x2 = tl.load(x2_ptrs, mask=offs < HALF_D)

    # Load cos and sin for this sequence position
    cos_ptrs = cos_ptr + pid_s * stride_cos_s + offs * stride_cos_d
    sin_ptrs = sin_ptr + pid_s * stride_sin_s + offs * stride_sin_d
    cos_val = tl.load(cos_ptrs, mask=offs < HALF_D)
    sin_val = tl.load(sin_ptrs, mask=offs < HALF_D)

    # Compute rotated halves
    out1 = x1 * cos_val - x2 * sin_val
    out2 = x1 * sin_val + x2 * cos_val

    # Store results
    tl.store(out_ptr + base_out + offs * stride_out_d, out1, mask=offs < HALF_D)
    tl.store(out_ptr + base_out + (offs + HALF_D) * stride_out_d, out2, mask=offs < HALF_D)


def triton_run(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    B, H, S, D = x.shape
    assert D == 128, "RoPE kernel assumes D=128"
    assert cos.shape == (S, D // 2) and sin.shape == (S, D // 2)
    assert x.dtype == torch.float16
    assert cos.dtype == torch.float16 and sin.dtype == torch.float16

    out = torch.empty_like(x)

    HALF_D = D // 2        # 64
    BLOCK_HALF = 64       # Load entire half-vector at once

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