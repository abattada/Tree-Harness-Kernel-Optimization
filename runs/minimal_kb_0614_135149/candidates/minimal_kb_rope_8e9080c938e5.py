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
    B,
    H,
    S,
    D: tl.constexpr,
    HALF_D: tl.constexpr,
):
    """
    RoPE kernel that processes all (batch, head) rows for a single sequence position.
    Reduces cos/sin memory traffic by factor B*H (256×), keeping them in registers.
    """
    s = tl.program_id(0)  # one program per sequence position, pid ∈ [0, S)
    offs_half = tl.arange(0, HALF_D)

    # Load cos/sin once for this s — reused across all b,h
    cos_vals = tl.load(cos_ptr + s * stride_cos_s + offs_half * stride_cos_d,
                       eviction_policy="evict_first")
    sin_vals = tl.load(sin_ptr + s * stride_sin_s + offs_half * stride_sin_d,
                       eviction_policy="evict_first")

    # Base offsets shared by all rows for this s
    base_x_s = s * stride_x_s
    base_out_s = s * stride_out_s

    # Loop over all batch and head positions
    total_bh = B * H
    for i in range(total_bh):
        b = i // H
        h = i % H

        # Pointers to the current row in x and out
        row_x = x_ptr + b * stride_x_b + h * stride_x_h + base_x_s
        row_out = out_ptr + b * stride_out_b + h * stride_out_h + base_out_s

        # Load two halves of the feature dimension
        x1 = tl.load(row_x + offs_half * stride_x_d, eviction_policy="evict_first")
        x2 = tl.load(row_x + (offs_half + HALF_D) * stride_x_d,
                     eviction_policy="evict_first")

        # RoPE arithmetic
        out1 = x1 * cos_vals - x2 * sin_vals
        out2 = x1 * sin_vals + x2 * cos_vals

        # Store results
        tl.store(row_out + offs_half * stride_out_d, out1)
        tl.store(row_out + (offs_half + HALF_D) * stride_out_d, out2)


def triton_run(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float16
    assert cos.dtype == torch.float16 and sin.dtype == torch.float16
    B, H, S, D = x.shape
    assert D == 128, "expected last dimension 128"
    assert cos.shape == (S, D // 2) and sin.shape == (S, D // 2), "cos/sin shape mismatch"

    out = torch.empty_like(x)

    HALF_D = D // 2  # 64

    # Launch one block per sequence position → each block processes all (B,H) rows
    grid = (S,)

    rope_kernel[grid](
        x, cos, sin, out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, S, D, HALF_D,
        num_warps=2,    # 64 threads exactly cover the 64-element half dimension
        num_stages=2,   # pipelining for the repeated loads
    )
    return out