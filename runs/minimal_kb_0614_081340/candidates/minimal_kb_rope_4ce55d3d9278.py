import torch
import triton
import triton.language as tl

@triton.jit
def rope_kernel_grouped(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    stride_x_b, stride_x_h, stride_x_s, stride_x_d,
    stride_cos_s, stride_cos_d,
    stride_sin_s, stride_sin_d,
    stride_out_b, stride_out_h, stride_out_s, stride_out_d,
    HALF_D: tl.constexpr,
):
    pid = tl.program_id(0)          # sequence position index
    if pid >= S:
        return

    s = pid

    # Base offsets for this sequence position
    s_off_x = s * stride_x_s
    s_off_out = s * stride_out_s
    s_off_cos = s * stride_cos_s
    s_off_sin = s * stride_sin_s

    # Load cos and sin values (shared across all (b,h) for this s)
    offs_half = tl.arange(0, HALF_D)
    cos_vals = tl.load(cos_ptr + s_off_cos + offs_half, mask=(offs_half < HALF_D), other=0.0)
    sin_vals = tl.load(sin_ptr + s_off_sin + offs_half, mask=(offs_half < HALF_D), other=0.0)

    # Process all B*H rows sharing this sequence position
    for b in range(B):
        b_off_x = b * stride_x_b
        b_off_out = b * stride_out_b
        for h in range(H):
            h_off_x = h * stride_x_h
            h_off_out = h * stride_out_h

            # Load two halves of x
            x1 = tl.load(x_ptr + b_off_x + h_off_x + s_off_x + offs_half)
            x2 = tl.load(x_ptr + b_off_x + h_off_x + s_off_x + offs_half + HALF_D)

            # Compute rotated halves
            y1 = x1 * cos_vals - x2 * sin_vals
            y2 = x1 * sin_vals + x2 * cos_vals

            # Store output
            out_base = out_ptr + b_off_out + h_off_out + s_off_out
            tl.store(out_base + offs_half, y1)
            tl.store(out_base + offs_half + HALF_D, y2)


@triton.jit
def rope_kernel_single_row(  # fallback for non‑standard sizes
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    B, H, S, D,
    stride_x_b, stride_x_h, stride_x_s, stride_x_d,
    stride_cos_s, stride_cos_d,
    stride_sin_s, stride_sin_d,
    stride_out_b, stride_out_h, stride_out_s, stride_out_d,
    BLOCK_D: tl.constexpr, HALF_D: tl.constexpr,
):
    pid = tl.program_id(0)
    total_rows = B * H * S
    if pid >= total_rows:
        return

    b = pid // (H * S)
    rem = pid % (H * S)
    h = rem // S
    s = rem % S

    # Base offsets
    x_base = x_ptr + b * stride_x_b + h * stride_x_h + s * stride_x_s
    out_base = out_ptr + b * stride_out_b + h * stride_out_h + s * stride_out_s

    offs_half = tl.arange(0, HALF_D)

    x1 = tl.load(x_base + offs_half)
    x2 = tl.load(x_base + offs_half + HALF_D)

    cos_ptr_s = cos_ptr + s * stride_cos_s
    sin_ptr_s = sin_ptr + s * stride_sin_s
    cos_vals = tl.load(cos_ptr_s + offs_half)
    sin_vals = tl.load(sin_ptr_s + offs_half)

    y1 = x1 * cos_vals - x2 * sin_vals
    y2 = x1 * sin_vals + x2 * cos_vals

    tl.store(out_base + offs_half, y1)
    tl.store(out_base + offs_half + HALF_D, y2)


def triton_run(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    B, H, S, D = x.shape
    assert D == 128, "RoPE kernel expects D=128"
    assert cos.shape == (S, D // 2), "cos shape mismatch"
    assert sin.shape == (S, D // 2), "sin shape mismatch"

    out = torch.empty_like(x)

    # Use grouped kernel when B*H fits comfortably and is not tiny
    if B * H >= 4:
        grid = (S,)
        rope_kernel_grouped[grid](
            x, cos, sin, out,
            B, H, S, D,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            cos.stride(0), cos.stride(1),
            sin.stride(0), sin.stride(1),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            HALF_D=64,
            num_warps=8,
            num_stages=3,
        )
    else:
        # Fallback to per‑row kernel for very small B*H
        grid = (B * H * S,)
        rope_kernel_single_row[grid](
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