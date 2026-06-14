import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr,
                 B, H, T, D, D2,
                 stride_xb, stride_xh, stride_xt, stride_xd,
                 stride_c_t, stride_c_d):
    """
    Each program computes one (b, h, t) row.
    x: [B, H, T, D]  f16, contiguous along last dim
    cos, sin: [T, D2] f16
    """
    pid = tl.program_id(0)
    # Decode linear row index back to (b, h, t)
    t = pid % T
    h_tmp = pid // T
    h = h_tmp % H
    b = h_tmp // H

    # Pointers to the start of the row in x and in cos/sin
    base_x = b * stride_xb + h * stride_xh + t * stride_xt
    base_c = t * stride_c_t

    # Load the two halves (64 elements each)
    x1 = tl.load(x_ptr + base_x + tl.arange(0, D2))
    x2 = tl.load(x_ptr + base_x + D2 + tl.arange(0, D2))
    c   = tl.load(cos_ptr + base_c + tl.arange(0, D2))
    s   = tl.load(sin_ptr + base_c + tl.arange(0, D2))

    # Fused rotate-half computation
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c

    # Store back as contiguous halves
    tl.store(out_ptr + base_x + tl.arange(0, D2), o1)
    tl.store(out_ptr + base_x + D2 + tl.arange(0, D2), o2)


def triton_run(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    RoPE operator for the given shapes and dtypes.
    x:   (8, 32, 4096, 128) f16
    cos: (4096, 64) f16
    sin: (4096, 64) f16
    """
    B, H, T, D = x.shape
    D2 = D // 2   # 64
    out = torch.empty_like(x)

    stride_xb, stride_xh, stride_xt, stride_xd = x.stride()
    stride_c_t, stride_c_d = cos.stride()

    num_rows = B * H * T               # 1,048,576
    grid = (num_rows,)                 # one program per row → high occupancy

    _rope_kernel[grid](
        x, cos, sin, out,
        B, H, T, D, D2,
        stride_xb, stride_xh, stride_xt, stride_xd,
        stride_c_t, stride_c_d,
        num_warps=4,   # 128 threads = 4 warps, fully using the 128-element row
        num_stages=2   # modest pipelining
    )
    return out