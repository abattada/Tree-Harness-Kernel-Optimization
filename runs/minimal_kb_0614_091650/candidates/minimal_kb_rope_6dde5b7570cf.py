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
    BLOCK_HALF: tl.constexpr, HALF_D: tl.constexpr,
):
    pid = tl.program_id(0)               # flat row index
    total_rows = B * H * S
    if pid >= total_rows:
        return

    # Decompose pid into (b, h, s)
    b = pid // (H * S)
    rem = pid % (H * S)
    h = rem // S
    s = rem % S

    # Base pointers for x and output
    x_base = x_ptr + b * stride_x_b + h * stride_x_h + s * stride_x_s
    out_base = out_ptr + b * stride_out_b + h * stride_out_h + s * stride_out_s

    # Offsets for the half dimension (0..63)
    offs = tl.arange(0, BLOCK_HALF)

    # Load the two halves of x
    x1 = tl.load(x_base + offs)                # first 64 elements
    x2 = tl.load(x_base + offs + HALF_D)       # second 64 elements

    # Load cos and sin for this sequence position
    cos_vals = tl.load(cos_ptr + s * stride_cos_s + offs)
    sin_vals = tl.load(sin_ptr + s * stride_sin_s + offs)

    # Compute rotated halves
    out1 = x1 * cos_vals - x2 * sin_vals
    out2 = x1 * sin_vals + x2 * cos_vals

    # Store output
    tl.store(out_base + offs, out1)
    tl.store(out_base + offs + HALF_D, out2)


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
        BLOCK_HALF=64, HALF_D=64,
        num_warps=8,
        num_stages=2,
    )

    return out