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
    """RoPE with vectorized, 128B-aligned loads via block pointers."""
    pid_bh = tl.program_id(0)
    pid_seq_block = tl.program_id(1)

    b = pid_bh // heads
    h = pid_bh % heads
    seq_block_start = pid_seq_block * BLOCK_SEQ

    # Base pointers for this (batch, head)
    base_x  = x_ptr  + b * stride_b + h * stride_h
    base_out = out_ptr + b * stride_b + h * stride_h

    # ---- x1: first half of feature dimension ----
    x1_block = tl.make_block_ptr(
        base=base_x,
        shape=(seq_len, d2),
        strides=(stride_s, stride_d),
        offsets=(seq_block_start, 0),
        block_shape=(BLOCK_SEQ, d2),
        order=(1, 0),          # row-major → contiguous along d2
    )
    # ---- x2: second half ----
    x2_block = tl.make_block_ptr(
        base=base_x + d2 * stride_d,
        shape=(seq_len, d2),
        strides=(stride_s, stride_d),
        offsets=(seq_block_start, 0),
        block_shape=(BLOCK_SEQ, d2),
        order=(1, 0),
    )

    # ---- cos, sin ----
    cos_block = tl.make_block_ptr(
        base=cos_ptr,
        shape=(seq_len, d2),
        strides=(stride_cos_s, stride_cos_d),
        offsets=(seq_block_start, 0),
        block_shape=(BLOCK_SEQ, d2),
        order=(1, 0),
    )
    sin_block = tl.make_block_ptr(
        base=sin_ptr,
        shape=(seq_len, d2),
        strides=(stride_sin_s, stride_sin_d),
        offsets=(seq_block_start, 0),
        block_shape=(BLOCK_SEQ, d2),
        order=(1, 0),
    )

    # Wide, coalesced loads with cache hints
    x1 = tl.load(x1_block, boundary_check=(0,), eviction_policy="evict_first")
    x2 = tl.load(x2_block, boundary_check=(0,), eviction_policy="evict_first")
    c  = tl.load(cos_block, boundary_check=(0,), eviction_policy="evict_last")
    s  = tl.load(sin_block, boundary_check=(0,), eviction_policy="evict_last")

    # ---- RoPE arithmetic ----
    out1 = x1 * c - x2 * s
    out2 = x1 * s + x2 * c

    # ---- store with block pointers ----
    out1_block = tl.make_block_ptr(
        base=base_out,
        shape=(seq_len, d2),
        strides=(stride_s, stride_d),
        offsets=(seq_block_start, 0),
        block_shape=(BLOCK_SEQ, d2),
        order=(1, 0),
    )
    out2_block = tl.make_block_ptr(
        base=base_out + d2 * stride_d,
        shape=(seq_len, d2),
        strides=(stride_s, stride_d),
        offsets=(seq_block_start, 0),
        block_shape=(BLOCK_SEQ, d2),
        order=(1, 0),
    )

    tl.store(out1_block, out1, boundary_check=(0,))
    tl.store(out2_block, out2, boundary_check=(0,))


def triton_run(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    RoPE (rotate-half) forward pass.
    x:   (batch, heads, seq_len, feat_dim)  float16
    cos: (seq_len, feat_dim//2)             float16
    sin: (seq_len, feat_dim//2)             float16
    Returns: (batch, heads, seq_len, feat_dim) float16
    """
    batch, heads, seq_len, feat_dim = x.shape
    d2 = feat_dim // 2

    assert cos.shape == (seq_len, d2), f"cos shape {cos.shape} != ({seq_len}, {d2})"
    assert sin.shape == (seq_len, d2), f"sin shape {sin.shape} != ({seq_len}, {d2})"

    out = torch.empty_like(x)

    # Strides (in elements, not bytes)
    stride_b, stride_h, stride_s, stride_d = x.stride()
    stride_cos_s, stride_cos_d = cos.stride()
    stride_sin_s, stride_sin_d = sin.stride()

    # Tuning knobs: a large BLOCK_SEQ amortizes launch overhead while
    # each row of 64 elements is exactly 128 bytes → coalesced & aligned.
    BLOCK_SEQ = 32
    num_warps = 8
    num_stages = 2

    grid = (batch * heads, triton.cdiv(seq_len, BLOCK_SEQ))

    rope_kernel[grid](
        x, cos, sin, out,
        batch=batch,
        heads=heads,
        seq_len=seq_len,
        d2=d2,
        stride_b=stride_b,
        stride_h=stride_h,
        stride_s=stride_s,
        stride_d=stride_d,
        stride_cos_s=stride_cos_s,
        stride_cos_d=stride_cos_d,
        stride_sin_s=stride_sin_s,
        stride_sin_d=stride_sin_d,
        BLOCK_SEQ=BLOCK_SEQ,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return out