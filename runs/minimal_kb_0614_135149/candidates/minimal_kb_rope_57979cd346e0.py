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
):
    """Each program handles one sequence position, reusing cos/sin across all batch and heads."""
    pid = tl.program_id(0)
    s = pid  # sequence index from 0..S-1

    offs_half = tl.arange(0, HALF_D)  # [0, 1, ..., 63]

    # Load cos and sin once for this sequence position, keep them live over the inner loops
    c = tl.load(cos_ptr + s * stride_cos_s + offs_half * stride_cos_d, eviction_policy='evict_last')
    s_vals = tl.load(sin_ptr + s * stride_sin_s + offs_half * stride_sin_d, eviction_policy='evict_last')

    for b in range(B):
        for h in range(H):
            # Base pointers for this (b, h, s) row
            base_x = x_ptr + b * stride_x_b + h * stride_x_h + s * stride_x_s
            base_out = out_ptr + b * stride_out_b + h * stride_out_h + s * stride_out_s

            # Load both halves of x (D = 128 -> each half is 64 fp16)
            x1 = tl.load(base_x + offs_half * stride_x_d, eviction_policy='evict_first')
            x2 = tl.load(base_x + (offs_half + HALF_D) * stride_x_d, eviction_policy='evict_first')

            # Rotate‑half formula
            out1 = x1 * c - x2 * s_vals
            out2 = x1 * s_vals + x2 * c

            # Store back the two halves
            tl.store(base_out + offs_half * stride_out_d, out1)
            tl.store(base_out + (offs_half + HALF_D) * stride_out_d, out2)


def triton_run(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings using the rotate‑half formulation.

    Args:
        x:   input  tensor, shape (B, H, S, 128), dtype float16
        cos: cosine table, shape (S, 64), dtype float16
        sin: sine   table, shape (S, 64), dtype float16
    Returns:
        output tensor, same shape and dtype as x.
    """
    assert x.dtype == torch.float16
    assert cos.dtype == torch.float16 and sin.dtype == torch.float16
    B, H, S, D = x.shape
    assert D == 128, f"expected last dim 128, got {D}"
    assert cos.shape == (S, D // 2) and sin.shape == (S, D // 2), "cos/sin shape mismatch"

    out = torch.empty_like(x)

    HALF_D = D // 2  # 64

    # One program per sequence position – cos/sin are loaded once and reused
    grid = (S,)

    rope_kernel[grid](
        x, cos, sin, out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B=B, H=H, S=S, D=D,
        HALF_D=HALF_D,
        num_warps=2,    # 64 threads cover the 64‑element half dimension exactly
        num_stages=2,   # enough to pipeline loads
    )
    return out