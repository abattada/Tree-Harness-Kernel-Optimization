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
    pid = tl.program_id(0)
    total_rows = B * H * S
    if pid >= total_rows:
        return

    # Decompose pid to (b, h, s)
    b = pid // (H * S)
    rem = pid % (H * S)
    h = rem // S
    s = rem % S

    # Base offsets for this row
    base_x = b * stride_x_b + h * stride_x_h + s * stride_x_s
    base_out = b * stride_out_b + h * stride_out_h + s * stride_out_s

    # Offsets for half dimension (0..63)
    offs_half = tl.arange(0, BLOCK_HALF)

    # Load first half x1 (contiguous)
    x1 = tl.load(
        x_ptr + base_x + offs_half * stride_x_d,
        eviction_policy='evict_first',
    )

    # Load second half x2 (contiguous, starts at offset 64)
    x2 = tl.load(
        x_ptr + base_x + (offs_half + HALF_D) * stride_x_d,
        eviction_policy='evict_first',
    )

    # Load cos and sin for this position (evict first since used once)
    cos_vals = tl.load(
        cos_ptr + s * stride_cos_s + offs_half * stride_cos_d,
        eviction_policy='evict_first',
    )
    sin_vals = tl.load(
        sin_ptr + s * stride_sin_s + offs_half * stride_sin_d,
        eviction_policy='evict_first',
    )

    # Compute rotated halves
    out1 = x1 * cos_vals - x2 * sin_vals
    out2 = x1 * sin_vals + x2 * cos_vals

    # Store results
    tl.store(
        out_ptr + base_out + offs_half * stride_out_d,
        out1,
    )
    tl.store(
        out_ptr + base_out + (offs_half + HALF_D) * stride_out_d,
        out2,
    )


def triton_run(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float16
    assert cos.dtype == torch.float16 and sin.dtype == torch.float16
    B, H, S, D = x.shape
    assert D == 128
    assert cos.shape == (S, D // 2) and sin.shape == (S, D // 2)

    out = torch.empty_like(x)

    HALF_D = D // 2          # 64
    BLOCK_HALF = 64          # load entire half at once

    # 1D grid over all rows
    grid = (B * H * S,)

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