import torch
import triton
import triton.language as tl


@triton.jit
def rope_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    batch: tl.constexpr,
    heads: tl.constexpr,
    seq_len: tl.constexpr,
    d2: tl.constexpr,
    stride_b: tl.constexpr,
    stride_h: tl.constexpr,
    stride_s: tl.constexpr,
    stride_d: tl.constexpr,
    stride_cos_s: tl.constexpr,
    stride_cos_d: tl.constexpr,
    stride_sin_s: tl.constexpr,
    stride_sin_d: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
):
    """Triton RoPE kernel with vectorized loads/stores and alignment hints."""
    pid_bh = tl.program_id(0)
    pid_seq_block = tl.program_id(1)

    b = pid_bh // heads
    h = pid_bh % heads

    seq_start = pid_seq_block * BLOCK_SEQ
    seq_offs = seq_start + tl.arange(0, BLOCK_SEQ)
    mask_seq = seq_offs < seq_len

    offs_d1 = tl.arange(0, d2)
    offs_d2 = tl.arange(d2, 2 * d2)

    base = b * stride_b + h * stride_h

    # Prepare address ranges
    x_offs_1 = base + seq_offs[:, None] * stride_s + offs_d1[None, :] * stride_d
    x_offs_2 = base + seq_offs[:, None] * stride_s + offs_d2[None, :] * stride_d
    cos_offs = seq_offs[:, None] * stride_cos_s + offs_d1[None, :] * stride_cos_d
    sin_offs = seq_offs[:, None] * stride_sin_s + offs_d1[None, :] * stride_sin_d

    # Vectorized loads with 128‑byte alignment hints
    x1 = tl.load(
        tl.multiple_of(x_ptr + x_offs_1, 128),
        mask=mask_seq[:, None],
        other=0.0,
    )
    x2 = tl.load(
        tl.multiple_of(x_ptr + x_offs_2, 128),
        mask=mask_seq[:, None],
        other=0.0,
    )

    cos_val = tl.load(
        tl.multiple_of(cos_ptr + cos_offs, 128),
        mask=mask_seq[:, None],
        other=0.0,
    )
    sin_val = tl.load(
        tl.multiple_of(sin_ptr + sin_offs, 128),
        mask=mask_seq[:, None],
        other=0.0,
    )

    # RoPE arithmetic
    out1 = x1 * cos_val - x2 * sin_val
    out2 = x1 * sin_val + x2 * cos_val

    # Vectorized stores with alignment hints
    tl.store(
        tl.multiple_of(out_ptr + x_offs_1, 128),
        out1,
        mask=mask_seq[:, None],
    )
    tl.store(
        tl.multiple_of(out_ptr + x_offs_2, 128),
        out2,
        mask=mask_seq[:, None],
    )


def triton_run(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """RoPE (rotate-half) forward – vectorized for high memory bandwidth."""
    batch, heads, seq_len, feat_dim = x.shape
    d2 = feat_dim // 2
    assert cos.shape == (seq_len, d2)
    assert sin.shape == (seq_len, d2)

    out = torch.empty_like(x)

    stride_b, stride_h, stride_s, stride_d = x.stride()
    stride_cos_s, stride_cos_d = cos.stride()
    stride_sin_s, stride_sin_d = sin.stride()

    # Tuned block size for better occupancy and vectorized width
    BLOCK_SEQ = 16
    num_warps = 8
    num_stages = 3

    grid = (batch * heads, triton.cdiv(seq_len, BLOCK_SEQ))

    rope_kernel[grid](
        x, cos, sin, out,
        batch, heads, seq_len, d2,
        stride_b, stride_h, stride_s, stride_d,
        stride_cos_s, stride_cos_d,
        stride_sin_s, stride_sin_d,
        BLOCK_SEQ=BLOCK_SEQ,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return out