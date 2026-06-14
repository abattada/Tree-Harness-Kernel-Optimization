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
    HALF_D: tl.constexpr,      # = D // 2 = 64
    BLOCK_HALF: tl.constexpr,  # = 64
):
    # 3D program grid (b, h, s)
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    # Base offsets for this row in x and out
    base_x   = pid_b * stride_x_b   + pid_h * stride_x_h   + pid_s * stride_x_s
    base_out = pid_b * stride_out_b + pid_h * stride_out_h + pid_s * stride_out_s

    # Offsets along the last dimension (0..63)
    offs = tl.arange(0, BLOCK_HALF)
    mask = offs < HALF_D          # always true if BLOCK_HALF == HALF_D

    # Load first half of x (x1)
    x1_ptrs = x_ptr + base_x + offs * stride_x_d
    x1 = tl.load(x1_ptrs, mask=mask)

    # Load second half of x (x2)
    x2_ptrs = x_ptr + base_x + (offs + HALF_D) * stride_x_d
    x2 = tl.load(x2_ptrs, mask=mask)

    # Load cos and sin for this sequence position
    cos_ptrs = cos_ptr + pid_s * stride_cos_s + offs * stride_cos_d
    sin_ptrs = sin_ptr + pid_s * stride_sin_s + offs * stride_sin_d
    c = tl.load(cos_ptrs, mask=mask)
    s = tl.load(sin_ptrs, mask=mask)

    # Compute rotated halves
    y1 = x1 * c - x2 * s
    y2 = x1 * s + x2 * c

    # Store first half
    out1_ptrs = out_ptr + base_out + offs * stride_out_d
    tl.store(out1_ptrs, y1, mask=mask)

    # Store second half
    out2_ptrs = out_ptr + base_out + (offs + HALF_D) * stride_out_d
    tl.store(out2_ptrs, y2, mask=mask)


def triton_run(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply rotate-half RoPE.
    x:  [B, H, S, D] float16, D=128
    cos: [S, D//2] float16
    sin: [S, D//2] float16
    Returns: [B, H, S, D] float16
    """
    B, H, S, D = x.shape
    assert D == 128, "RoPE kernel expects D=128"
    assert cos.shape == (S, D // 2), "cos shape mismatch"
    assert sin.shape == (S, D // 2), "sin shape mismatch"
    assert x.dtype == torch.float16
    assert cos.dtype == torch.float16 and sin.dtype == torch.float16

    out = torch.empty_like(x)

    HALF_D = D // 2  # 64
    BLOCK_HALF = 64

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