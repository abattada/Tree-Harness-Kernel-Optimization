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
    BLOCK: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    # Base pointers for the row in x and out
    x_base = x_ptr + pid_b * stride_x_b + pid_h * stride_x_h + pid_s * stride_x_s
    out_base = out_ptr + pid_b * stride_out_b + pid_h * stride_out_h + pid_s * stride_out_s

    # Offsets within the half dimension
    offs = tl.arange(0, BLOCK)  # 0 .. 63

    # Load first half of x (x1)
    x1 = tl.load(x_base + offs * stride_x_d)  # BLOCK = HALF_D, so no mask needed

    # Load second half of x (x2)
    x2 = tl.load(x_base + (offs + HALF_D) * stride_x_d)

    # Load cos and sin for the current sequence position
    cos_base = cos_ptr + pid_s * stride_cos_s
    sin_base = sin_ptr + pid_s * stride_sin_s
    c = tl.load(cos_base + offs * stride_cos_d)
    s = tl.load(sin_base + offs * stride_sin_d)

    # Compute rotated halves
    out1 = x1 * c - x2 * s
    out2 = x1 * s + x2 * c

    # Store results
    tl.store(out_base + offs * stride_out_d, out1)
    tl.store(out_base + (offs + HALF_D) * stride_out_d, out2)


def triton_run(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotate-half RoPE.

    Args:
        x: (B, H, S, D) float16, D must be 128.
        cos: (S, D//2) float16
        sin: (S, D//2) float16

    Returns:
        out: (B, H, S, D) float16, same as reference
    """
    B, H, S, D = x.shape
    assert D == 128, f"RoPE kernel requires D=128, got D={D}"
    assert cos.shape == (S, D // 2) and sin.shape == (S, D // 2)
    assert x.dtype == torch.float16 and cos.dtype == torch.float16 and sin.dtype == torch.float16

    out = torch.empty_like(x)

    HALF_D = D // 2  # 64
    BLOCK = HALF_D    # load entire half in one go

    grid = (B, H, S)

    rope_kernel[grid](
        x, cos, sin, out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, S, D,
        HALF_D, BLOCK,
        num_warps=4,
        num_stages=2,
    )

    return out