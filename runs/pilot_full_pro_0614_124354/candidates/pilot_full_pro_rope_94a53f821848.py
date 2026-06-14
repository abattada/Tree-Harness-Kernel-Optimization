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
    ROWS_PER_PROG: tl.constexpr,
):
    """
    RoPE forward with multirow processing.
    Each program handles ROWS_PER_PROG consecutive sequence positions for
    a single (batch, head) pair, amortising launch overhead and improving
    memory access efficiency on the short feature dimension (d2=64).
    """
    pid_bh = tl.program_id(0)
    pid_seq_block = tl.program_id(1)

    b = pid_bh // heads
    h = pid_bh % heads

    # Sequence positions handled by this program
    seq_start = pid_seq_block * ROWS_PER_PROG
    seq_offs = seq_start + tl.arange(0, ROWS_PER_PROG)
    mask_seq = seq_offs < seq_len

    # Feature dimension ranges (d2 elements each half)
    offs_d1 = tl.arange(0, d2)
    offs_d2 = tl.arange(d2, 2 * d2)

    # Base offset for batch + head
    base = b * stride_b + h * stride_h

    # ---- load x1 and x2 with streaming hint ----
    x_offs_1 = base + seq_offs[:, None] * stride_s + offs_d1[None, :] * stride_d
    x_offs_2 = base + seq_offs[:, None] * stride_s + offs_d2[None, :] * stride_d

    x1 = tl.load(x_ptr + x_offs_1, mask=mask_seq[:, None], other=0.0,
                 eviction_policy='evict_first')
    x2 = tl.load(x_ptr + x_offs_2, mask=mask_seq[:, None], other=0.0,
                 eviction_policy='evict_first')

    # ---- load cos and sin (same eviction hint) ----
    cos_offs = seq_offs[:, None] * stride_cos_s + offs_d1[None, :] * stride_cos_d
    sin_offs = seq_offs[:, None] * stride_sin_s + offs_d1[None, :] * stride_sin_d

    cos_val = tl.load(cos_ptr + cos_offs, mask=mask_seq[:, None], other=0.0,
                      eviction_policy='evict_first')
    sin_val = tl.load(sin_ptr + sin_offs, mask=mask_seq[:, None], other=0.0,
                      eviction_policy='evict_first')

    # ---- RoPE arithmetic ----
    out1 = x1 * cos_val - x2 * sin_val
    out2 = x1 * sin_val + x2 * cos_val

    # ---- store (streaming write, no reuse) ----
    tl.store(out_ptr + x_offs_1, out1, mask=mask_seq[:, None],
             eviction_policy='evict_first')
    tl.store(out_ptr + x_offs_2, out2, mask=mask_seq[:, None],
             eviction_policy='evict_first')


def triton_run(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    RoPE (rotate-half) forward pass, multirow variant.
    x:   (batch, heads, seq_len, feat_dim)  float16
    cos: (seq_len, feat_dim//2)             float16
    sin: (seq_len, feat_dim//2)             float16
    Returns: (batch, heads, seq_len, feat_dim) float16
    """
    batch, heads, seq_len, feat_dim = x.shape
    d2 = feat_dim // 2

    assert cos.shape == (seq_len, d2)
    assert sin.shape == (seq_len, d2)

    out = torch.empty_like(x)

    stride_b, stride_h, stride_s, stride_d = x.stride()
    stride_cos_s, stride_cos_d = cos.stride()
    stride_sin_s, stride_sin_d = sin.stride()

    # Tuned for RTX 5090: 32 rows per program gives large tiles that saturate BW.
    ROWS_PER_PROG = 32       # balanced to minimise launch overhead and maximise coalescing
    num_warps = 8            # enough warps to hide instruction latency without excessive reg pressure
    num_stages = 4           # pipeline the loads from L2

    grid = (batch * heads, triton.cdiv(seq_len, ROWS_PER_PROG))

    rope_kernel[grid](
        x, cos, sin, out,
        batch, heads, seq_len, d2,
        stride_b, stride_h, stride_s, stride_d,
        stride_cos_s, stride_cos_d,
        stride_sin_s, stride_sin_d,
        ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return out