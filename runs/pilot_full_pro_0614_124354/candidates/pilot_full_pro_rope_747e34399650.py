import torch
import triton
import triton.language as tl


@triton.jit
def rope_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    B: int,
    H: int,
    S: int,
    stride_xb: int,
    stride_xh: int,
    stride_xs: int,
    stride_cos_s: int,
    stride_cos_d: int,
    stride_sin_s: int,
    stride_sin_d: int,
    BLOCK_D: tl.constexpr,
    BLOCK_D2: tl.constexpr,
):
    """Apply rotary embeddings: each program handles one (b, h, s) row of length D."""
    pid = tl.program_id(0)

    # map 1D program id to (b, h, s)
    HS = H * S
    b = pid // HS
    rem = pid % HS
    h = rem // S
    s = rem % S

    # base pointers for this row
    off_x = b * stride_xb + h * stride_xh + s * stride_xs
    off_cos = s * stride_cos_s
    off_sin = s * stride_sin_s

    # load entire row of x (128 elements) and the corresponding cos/sin (64 elements each)
    x_row = tl.load(x_ptr + off_x + tl.arange(0, BLOCK_D))
    cos_row = tl.load(cos_ptr + off_cos + tl.arange(0, BLOCK_D2))
    sin_row = tl.load(sin_ptr + off_sin + tl.arange(0, BLOCK_D2))

    # split x into the two halves
    x_reshaped = tl.reshape(x_row, [2, BLOCK_D2])
    x1 = x_reshaped[0, :]  # first  64 elements
    x2 = x_reshaped[1, :]  # second 64 elements

    # rotated halves
    out1 = x1 * cos_row - x2 * sin_row
    out2 = x1 * sin_row + x2 * cos_row

    # store the two halves back into the output row
    out_off = b * stride_xb + h * stride_xh + s * stride_xs  # same strides as x
    tl.store(out_ptr + out_off + tl.arange(0, BLOCK_D2), out1)
    tl.store(out_ptr + out_off + BLOCK_D2 + tl.arange(0, BLOCK_D2), out2)


def triton_run(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    RoPE (rotate-half) as defined in the reference.
    x:   (B, H, S, D)   with D = 128
    cos: (S, D//2)
    sin: (S, D//2)
    returns: (B, H, S, D)
    """
    B, H, S, D = x.shape
    D2 = D // 2
    assert D == 128, "Last dim must be 128"
    assert cos.shape == (S, D2), f"cos shape mismatch: {cos.shape} vs {(S, D2)}"
    assert sin.shape == (S, D2), f"sin shape mismatch: {sin.shape} vs {(S, D2)}"

    out = torch.empty_like(x)

    # Launch one program per (b, h, s) row
    grid = (B * H * S,)

    rope_kernel[grid](
        x,
        cos,
        sin,
        out,
        B,
        H,
        S,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        cos.stride(0),
        cos.stride(1),
        sin.stride(0),
        sin.stride(1),
        BLOCK_D=128,
        BLOCK_D2=64,
        num_warps=4,
        num_stages=2,
    )
    return out