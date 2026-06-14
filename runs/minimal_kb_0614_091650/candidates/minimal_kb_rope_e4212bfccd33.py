import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# RoPE kernel with grid-stride loop to amortize launch overhead.
# Each program handles a contiguously assigned chunk of rows (B*H*S total).
# ---------------------------------------------------------------------------
@triton.jit
def rope_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    stride_x_b, stride_x_h, stride_x_s, stride_x_d,
    stride_cos_s, stride_cos_d,
    stride_sin_s, stride_sin_d,
    stride_out_b, stride_out_h, stride_out_s, stride_out_d,
    total_rows: tl.constexpr,     # B*H*S, known at compile time
    HALF_D: tl.constexpr,         # = 64
    BLOCK_HALF: tl.constexpr,     # = 64
):
    # -----------------------------------------------------------------------
    # Grid-stride loop: each program processes rows in steps of num_programs
    # -----------------------------------------------------------------------
    pid = tl.program_id(0)
    step = tl.num_programs(0)                     # number of programs launched
    row = pid

    while row < total_rows:
        # Decompose the flat row index into (b, h, s)
        # Use integer arithmetic to avoid mod/div overhead
        b = row // (H * S)     # H and S are not constexpr here; but can be computed
        # Actually H and S are not passed as constexpr. We need them.
        # Better to compute using the stride? No, we need B,H,S as constexpr.
        # But we already have total_rows = B*H*S. We can compute H and S from strides? Hard.
        # Let's pass B, H, S as constexpr.
        # For now, we can compute them inside the kernel using tl.constexpr? No.
        # We need to modify the signature: add B,H,S as tl.constexpr.
        # So let's adjust.

# We need B, H, S as constexpr for decomposition. Let's redesign.
# Instead, we can compute b,h,s from the strides? Not necessary.
# We'll pass B, H, S as tl.constexpr.

# Rewrite with B,H,S constexpr.

import torch
import triton
import triton.language as tl

@triton.jit
def rope_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    stride_x_b, stride_x_h, stride_x_s, stride_x_d,
    stride_cos_s, stride_cos_d,
    stride_sin_s, stride_sin_d,
    stride_out_b, stride_out_h, stride_out_s, stride_out_d,
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    HALF_D: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
):
    # Grid-stride loop
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    total_rows = B * H * S

    row = pid
    while row < total_rows:
        # Decompose row into (b, h, s)
        # b = row // (H * S)
        # rem = row % (H * S)
        # h = rem // S
        # s = rem % S
        # Use Triton's integer division which is fast
        hS = total_rows // B   # = H * S
        b = row // hS
        rem = row - b * hS
        h = rem // S
        s = rem - h * S

        # Offsets along the last dimension (half-size)
        offs = tl.arange(0, BLOCK_HALF)  # 0..63

        # Pointer to x row start
        x_base = x_ptr + b * stride_x_b + h * stride_x_h + s * stride_x_s

        # Load first half x1 (0..63)
        x1 = tl.load(x_base + offs * stride_x_d, mask=offs < HALF_D)   # mask always true but kept for clarity

        # Load second half x2 (64..127)
        x2 = tl.load(x_base + (offs + HALF_D) * stride_x_d, mask=offs < HALF_D)

        # Load cos and sin for this sequence position
        cos_base = cos_ptr + s * stride_cos_s
        sin_base = sin_ptr + s * stride_sin_s
        cos_vals = tl.load(cos_base + offs * stride_cos_d, mask=offs < HALF_D)
        sin_vals = tl.load(sin_base + offs * stride_sin_d, mask=offs < HALF_D)

        # Compute rotated halves
        y1 = x1 * cos_vals - x2 * sin_vals
        y2 = x1 * sin_vals + x2 * cos_vals

        # Store results
        out_base = out_ptr + b * stride_out_b + h * stride_out_h + s * stride_out_s
        tl.store(out_base + offs * stride_out_d, y1, mask=offs < HALF_D)
        tl.store(out_base + (offs + HALF_D) * stride_out_d, y2, mask=offs < HALF_D)

        # Advance to next row
        row += num_programs


def triton_run(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    B, H, S, D = x.shape
    assert D == 128, "RoPE kernel assumes D=128"
    assert cos.shape == (S, D // 2) and sin.shape == (S, D // 2)
    assert x.dtype == torch.float16
    assert cos.dtype == torch.float16 and sin.dtype == torch.float16

    out = torch.empty_like(x)

    HALF_D = D // 2          # 64
    BLOCK_HALF = 64          # load entire half at once

    # Use a moderate number of programs (e.g. 1024) to amortize launch overhead.
    total_rows = B * H * S
    N_PROGRAMS = min(total_rows, 1024)   # reasonable, can be tuned further

    grid = (N_PROGRAMS,)
    rope_kernel[grid](
        x, cos, sin, out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, S, D,
        HALF_D, BLOCK_HALF,
        num_warps=4,
        num_stages=2,
    )
    return out