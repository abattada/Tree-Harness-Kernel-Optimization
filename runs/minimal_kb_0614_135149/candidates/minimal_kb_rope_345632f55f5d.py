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
    ROWS_PER_PROG: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_s_block = tl.program_id(1)

    # Decompose batch-head index into b, h (only once per program)
    b = pid_bh // H
    h = pid_bh % H

    start_s = pid_s_block * ROWS_PER_PROG

    # Precompute batch+head offsets
    base_bh_x = b * stride_x_b + h * stride_x_h
    base_bh_out = b * stride_out_b + h * stride_out_h

    offs_half = tl.arange(0, HALF_D)

    for i in range(ROWS_PER_PROG):
        s = start_s + i

        # Base pointers for this row
        x_base = base_bh_x + s * stride_x_s
        out_base = base_bh_out + s * stride_out_s

        # Load x halves (contiguous access, no masks)
        x1 = tl.load(x_ptr + x_base + offs_half * stride_x_d, eviction_policy='evict_first')
        x2 = tl.load(x_ptr + x_base + (offs_half + HALF_D) * stride_x_d, eviction_policy='evict_first')

        # Load cos and sin for this position
        cos_base = s * stride_cos_s
        sin_base = s * stride_sin_s
        c = tl.load(cos_ptr + cos_base + offs_half * stride_cos_d, eviction_policy='evict_first')
        s_val = tl.load(sin_ptr + sin_base + offs_half * stride_sin_d, eviction_policy='evict_first')

        # Compute rotated halves
        out1 = x1 * c - x2 * s_val
        out2 = x1 * s_val + x2 * c

        # Store
        tl.store(out_ptr + out_base + offs_half * stride_out_d, out1)
        tl.store(out_ptr + out_base + (offs_half + HALF_D) * stride_out_d, out2)


def triton_run(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float16
    assert cos.dtype == torch.float16 and sin.dtype == torch.float16
    B, H, S, D = x.shape
    assert D == 128
    assert cos.shape == (S, D // 2) and sin.shape == (S, D // 2)

    out = torch.empty_like(x)

    HALF_D = D // 2  # 64
    ROWS_PER_PROG = 32
    assert S % ROWS_PER_PROG == 0, "S must be divisible by ROWS_PER_PROG"

    # 2‑D grid: one dimension per (batch, head) pair, the other over blocks of S
    grid = (B * H, S // ROWS_PER_PROG)

    rope_kernel[grid](
        x, cos, sin, out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, S, D,
        HALF_D,
        ROWS_PER_PROG,
        num_warps=8,
        num_stages=3,
    )
    return out