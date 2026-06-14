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
    RoPE forward kernel. Each program handles a tile of BLOCK_SEQ sequence positions
    for one (batch, head) pair.
    """
    # Compile-time hint: feature dimension should be a multiple of 16 so the compiler
    # can emit wide vectorized loads.
    tl.static_assert(d2 % 16 == 0, "d2 must be a multiple of 16 for vectorized access")

    pid_bh = tl.program_id(0)
    pid_seq_block = tl.program_id(1)

    b = pid_bh // heads
    h = pid_bh % heads

    seq_start = pid_seq_block * BLOCK_SEQ
    seq_offs = seq_start + tl.arange(0, BLOCK_SEQ)
    mask_seq = seq_offs < seq_len

    # Hint that seq_start is a multiple of BLOCK_SEQ – helps the compiler optimise
    # modulo/division operations in address calculations.
    seq_start = tl.multiple_of(seq_start, BLOCK_SEQ)

    offs_d1 = tl.arange(0, d2)           # first half of feature dim
    offs_d2 = tl.arange(d2, 2 * d2)      # second half

    base = b * stride_b + h * stride_h

    # Compute linear indices for x1 and x2
    x_offs_1 = base + seq_offs[:, None] * stride_s + offs_d1[None, :] * stride_d
    x_offs_2 = base + seq_offs[:, None] * stride_s + offs_d2[None, :] * stride_d

    # Load two halves of the feature vector
    x1 = tl.load(x_ptr + x_offs_1, mask=mask_seq[:, None], other=0.0)
    x2 = tl.load(x_ptr + x_offs_2, mask=mask_seq[:, None], other=0.0)

    # Load cos and sin values for these sequence positions
    cos_offs = seq_offs[:, None] * stride_cos_s + offs_d1[None, :] * stride_cos_d
    sin_offs = seq_offs[:, None] * stride_sin_s + offs_d1[None, :] * stride_sin_d

    cos_val = tl.load(cos_ptr + cos_offs, mask=mask_seq[:, None], other=0.0)
    sin_val = tl.load(sin_ptr + sin_offs, mask=mask_seq[:, None], other=0.0)

    # Apply RoPE transformation
    out1 = x1 * cos_val - x2 * sin_val
    out2 = x1 * sin_val + x2 * cos_val

    # Store the rotated halves back to their original positions
    tl.store(out_ptr + x_offs_1, out1, mask=mask_seq[:, None])
    tl.store(out_ptr + x_offs_2, out2, mask=mask_seq[:, None])


def triton_run(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Public entry point: allocates output, launches the kernel, returns the result.
    Signature: triton_run(x, cos, sin) -> out
    """
    batch, heads, seq_len, feat_dim = x.shape
    d2 = feat_dim // 2

    # Sanity checks (same as reference)
    assert cos.shape == (seq_len, d2), f"cos shape mismatch"
    assert sin.shape == (seq_len, d2), f"sin shape mismatch"

    out = torch.empty_like(x)

    stride_b, stride_h, stride_s, stride_d = x.stride()
    stride_cos_s, stride_cos_d = cos.stride()
    stride_sin_s, stride_sin_d = sin.stride()

    # Tuned launch parameters (kept from the fast parent version)
    BLOCK_SEQ = 4
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