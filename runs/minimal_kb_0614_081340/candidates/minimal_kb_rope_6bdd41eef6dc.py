import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel: Rotary Position Embedding (RoPE) with multi-row processing
# Each program processes ROWS_PER_BLOCK consecutive sequence positions (rows)
# to achieve larger contiguous memory accesses and higher occupancy.
# ---------------------------------------------------------------------------

@triton.jit
def rope_kernel_multirow(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    B, H, S, D,
    stride_x_b, stride_x_h, stride_x_s, stride_x_d,
    stride_cos_s, stride_cos_d,
    stride_sin_s, stride_sin_d,
    stride_out_b, stride_out_h, stride_out_s, stride_out_d,
    ROWS_PER_BLOCK: tl.constexpr, HALF_D: tl.constexpr,
):
    # Flat program id decomposed into (batch, head, block of rows)
    pid_h = tl.program_id(0)          # 0 .. B*H*num_blocks_s - 1
    num_blocks_s = tl.cdiv(S, ROWS_PER_BLOCK)

    # Recover indices
    b = pid_h // (H * num_blocks_s)
    rem_h = pid_h % (H * num_blocks_s)
    h = rem_h // num_blocks_s
    s_block = rem_h % num_blocks_s

    s_start = s_block * ROWS_PER_BLOCK

    # Row offsets within the block (0..ROWS_PER_BLOCK-1)
    offs_row = tl.arange(0, ROWS_PER_BLOCK)
    # Column offsets along D dimension (0..D-1) and its halved version
    offs_d = tl.arange(0, D)
    offs_half = tl.arange(0, HALF_D)

    # Compute base pointers for x, cos, sin, out for this (b,h,s_start)
    x_base = x_ptr + b * stride_x_b + h * stride_x_h + s_start * stride_x_s
    cos_base = cos_ptr + s_start * stride_cos_s
    sin_base = sin_ptr + s_start * stride_sin_s
    out_base = out_ptr + b * stride_out_b + h * stride_out_h + s_start * stride_out_s

    # Mask for rows that are within the sequence length
    row_mask = s_start + offs_row < S
    # 2D mask for loads/stores (broadcast over D)
    mask_2d = row_mask[:, None] & (offs_d[None, :] < D)   # always true for D-dim

    # Load the full x tile of shape (ROWS_PER_BLOCK, D) using a single load
    # Memory layout: consecutive rows of size D, so contiguous in memory.
    x = tl.load(x_base + offs_row[:, None] * stride_x_s + offs_d[None, :],
                mask=mask_2d, other=0.0).to(tl.float16)

    # Split into two halves along the last dimension
    offs_half_2d = offs_half[None, :]
    x1 = tl.load(x_base + offs_row[:, None] * stride_x_s + offs_half_2d,
                 mask=row_mask[:, None], other=0.0).to(tl.float16)
    x2 = tl.load(x_base + offs_row[:, None] * stride_x_s + (offs_half_2d + HALF_D),
                 mask=row_mask[:, None], other=0.0).to(tl.float16)

    # Load cos and sin for these rows (size ROWS_PER_BLOCK x HALF_D)
    # cos/sin are contiguous across rows (stride_cos_s = HALF_D)
    cos_vals = tl.load(cos_base + offs_row[:, None] * stride_cos_s + offs_half_2d,
                       mask=row_mask[:, None], other=0.0).to(tl.float16)
    sin_vals = tl.load(sin_base + offs_row[:, None] * stride_sin_s + offs_half_2d,
                       mask=row_mask[:, None], other=0.0).to(tl.float16)

    # Compute rotated halves
    y1 = x1 * cos_vals - x2 * sin_vals
    y2 = x1 * sin_vals + x2 * cos_vals

    # Store the results (two halves)
    tl.store(out_base + offs_row[:, None] * stride_out_s + offs_half_2d, y1,
             mask=row_mask[:, None])
    tl.store(out_base + offs_row[:, None] * stride_out_s + (offs_half_2d + HALF_D), y2,
             mask=row_mask[:, None])


def triton_run(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    B, H, S, D = x.shape
    assert D == 128, "RoPE kernel expects D=128"
    assert cos.shape == (S, D // 2), "cos shape mismatch"
    assert sin.shape == (S, D // 2), "sin shape mismatch"

    out = torch.empty_like(x)

    # Tuned parameters for this hardware
    ROWS_PER_BLOCK = 4
    HALF_D = D // 2

    num_blocks_s = (S + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK
    grid = (B * H * num_blocks_s,)

    rope_kernel_multirow[grid](
        x, cos, sin, out,
        B, H, S, D,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        ROWS_PER_BLOCK=ROWS_PER_BLOCK, HALF_D=HALF_D,
        num_warps=16,              # 16 warps * 32 threads = 512 threads (full block)
        num_stages=3,
    )
    return out