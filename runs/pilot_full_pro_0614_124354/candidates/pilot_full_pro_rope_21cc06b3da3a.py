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
    """
    RoPE forward kernel with static shape specialization.
    Each program handles BLOCK_SEQ sequence positions for one (batch, head) pair.
    All masks are eliminated because seq_len is a multiple of BLOCK_SEQ.
    """
    # Compile-time assertion: perfect divisibility allows mask-free access
    tl.static_assert(seq_len % BLOCK_SEQ == 0, "seq_len must be a multiple of BLOCK_SEQ")

    pid_bh = tl.program_id(0)
    pid_seq_block = tl.program_id(1)

    b = pid_bh // heads
    h = pid_bh % heads

    seq_start = pid_seq_block * BLOCK_SEQ
    seq_offs = seq_start + tl.arange(0, BLOCK_SEQ)

    # Feature dimension halves: d2 known at compile time
    offs_d1 = tl.arange(0, d2)          # first half: 0..d2-1
    offs_d2 = tl.arange(d2, 2 * d2)     # second half: d2..2*d2-1

    base = b * stride_b + h * stride_h

    # ---- load x1, x2 ----
    x_offs_1 = base + seq_offs[:, None] * stride_s + offs_d1[None, :] * stride_d
    x_offs_2 = base + seq_offs[:, None] * stride_s + offs_d2[None, :] * stride_d

    # Hint: max_contiguous = d2 tells compiler the innermost dimension is dense
    x1 = tl.load(x_ptr + x_offs_1, other=0.0, max_contiguous=d2)
    x2 = tl.load(x_ptr + x_offs_2, other=0.0, max_contiguous=d2)

    # ---- load cos, sin ----
    cos_offs = seq_offs[:, None] * stride_cos_s + offs_d1[None, :] * stride_cos_d
    sin_offs = seq_offs[:, None] * stride_sin_s + offs_d1[None, :] * stride_sin_d

    cos_val = tl.load(cos_ptr + cos_offs, other=0.0, max_contiguous=d2)
    sin_val = tl.load(sin_ptr + sin_offs, other=0.0, max_contiguous=d2)

    # ---- RoPE computation ----
    out1 = x1 * cos_val - x2 * sin_val
    out2 = x1 * sin_val + x2 * cos_val

    # ---- store results mask‑free ----
    tl.store(out_ptr + x_offs_1, out1)
    tl.store(out_ptr + x_offs_2, out2)


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

    stride_b, stride_h, stride_s, stride_d = x.stride()
    stride_cos_s, stride_cos_d = cos.stride()
    stride_sin_s, stride_sin_d = sin.stride()

    # Tuning: BLOCK_SEQ must divide seq_len to keep the mask‑free fast path.
    BLOCK_SEQ = 8   # 4096 % 8 == 0, reduces launch count vs. parent's 4
    num_warps = 4
    num_stages = 2

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