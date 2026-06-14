import torch
import triton
import triton.language as tl

@triton.jit
def rope_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    B, H, S, D,
    stride_x_b, stride_x_h, stride_x_s, stride_x_d,
    stride_cos_s, stride_cos_d,
    stride_sin_s, stride_sin_d,
    stride_out_b, stride_out_h, stride_out_s, stride_out_d,
    BLOCK_D: tl.constexpr, HALF_D: tl.constexpr,
):
    pid = tl.program_id(0)              # flat row index
    total_rows = B * H * S
    if pid >= total_rows:
        return

    # Decompose pid into (b, h, s)
    b = pid // (H * S)
    rem = pid % (H * S)
    h = rem // S
    s = rem % S

    # Base pointer for x/out: row start
    x_base = x_ptr + b * stride_x_b + h * stride_x_h + s * stride_x_s
    out_base = out_ptr + b * stride_out_b + h * stride_out_h + s * stride_out_s

    # Offsets along D
    offs_half = tl.arange(0, HALF_D)

    # Load first and second halves of x (each 64 elements)
    x1 = tl.load(x_base + offs_half)   # first 64
    x2 = tl.load(x_base + (offs_half + HALF_D))  # second 64

    # Load cos and sin for this sequence position
    cos_ptr_s = cos_ptr + s * stride_cos_s
    sin_ptr_s = sin_ptr + s * stride_sin_s
    cos_vals = tl.load(cos_ptr_s + offs_half)   # 64 elements
    sin_vals = tl.load(sin_ptr_s + offs_half)   # 64 elements

    # Compute rotated halves
    y1 = x1 * cos_vals - x2 * sin_vals
    y2 = x1 * sin_vals + x2 * cos_vals

    # Store output
    tl.store(out_base + offs_half, y1)
    tl.store(out_base + offs_half + HALF_D, y2)


def triton_run(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    B, H, S, D = x.shape
    assert D == 128, "RoPE kernel expects D=128"
    assert cos.shape == (S, D // 2), "cos shape mismatch"
    assert sin.shape == (S, D // 2), "sin shape mismatch"

    out = torch.empty_like(x)

    # Launch grid = total number of rows
    grid = (B * H * S,)
    rope_kernel[grid](
        x, cos, sin, out,
        B, H, S, D,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_D=128, HALF_D=64,
        num_warps=4,
        num_stages=2,
    )
    return out