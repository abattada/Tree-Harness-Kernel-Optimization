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
    BLOCK_SEQ: tl.constexpr,
):
    """
    RoPE forward pass with multi‑row processing.
    Each program handles ROWS_PER_PROG batch*head rows and BLOCK_SEQ sequence
    positions, looping over the rows sequentially to keep register pressure low.
    """
    pid_rows = tl.program_id(0)
    pid_seq = tl.program_id(1)

    # which rows this program is responsible for
    row_start = pid_rows * ROWS_PER_PROG
    total_rows = batch * heads

    # sequence block
    seq_start = pid_seq * BLOCK_SEQ
    seq_offs = seq_start + tl.arange(0, BLOCK_SEQ)
    mask_seq = seq_offs < seq_len

    # feature dimension halves
    off_d1 = tl.arange(0, d2)
    off_d2 = tl.arange(d2, 2 * d2)

    for r in range(ROWS_PER_PROG):
        row = row_start + r
        if row >= total_rows:
            continue

        b = row // heads
        h = row % heads

        base = b * stride_b + h * stride_h

        # ---- load x1, x2 ----
        x_offs_1 = base + seq_offs[:, None] * stride_s + off_d1[None, :] * stride_d
        x_offs_2 = base + seq_offs[:, None] * stride_s + off_d2[None, :] * stride_d

        x1 = tl.load(x_ptr + x_offs_1, mask=mask_seq[:, None], other=0.0)
        x2 = tl.load(x_ptr + x_offs_2, mask=mask_seq[:, None], other=0.0)

        # ---- load cos, sin ----
        cos_offs = seq_offs[:, None] * stride_cos_s + off_d1[None, :] * stride_cos_d
        sin_offs = seq_offs[:, None] * stride_sin_s + off_d1[None, :] * stride_sin_d

        cos_val = tl.load(cos_ptr + cos_offs, mask=mask_seq[:, None], other=0.0)
        sin_val = tl.load(sin_ptr + sin_offs, mask=mask_seq[:, None], other=0.0)

        # ---- RoPE arithmetic ----
        out1 = x1 * cos_val - x2 * sin_val
        out2 = x1 * sin_val + x2 * cos_val

        # ---- store ----
        tl.store(out_ptr + x_offs_1, out1, mask=mask_seq[:, None])
        tl.store(out_ptr + x_offs_2, out2, mask=mask_seq[:, None])


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

    # Strides
    stride_b, stride_h, stride_s, stride_d = x.stride()
    stride_cos_s, stride_cos_d = cos.stride()
    stride_sin_s, stride_sin_d = sin.stride()

    # ---- tuning knobs (multi‑row variant) ----
    ROWS_PER_PROG = 8   # batch*head rows per program
    BLOCK_SEQ = 8       # sequence positions per program block
    num_warps = 8
    num_stages = 2

    # Grid: (ceil(num_rows / ROWS_PER_PROG), ceil(seq_len / BLOCK_SEQ))
    total_rows = batch * heads
    grid = (triton.cdiv(total_rows, ROWS_PER_PROG), triton.cdiv(seq_len, BLOCK_SEQ))

    rope_kernel[grid](
        x, cos, sin, out,
        batch, heads, seq_len, d2,
        stride_b, stride_h, stride_s, stride_d,
        stride_cos_s, stride_cos_d,
        stride_sin_s, stride_sin_d,
        ROWS_PER_PROG=ROWS_PER_PROG,
        BLOCK_SEQ=BLOCK_SEQ,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return out