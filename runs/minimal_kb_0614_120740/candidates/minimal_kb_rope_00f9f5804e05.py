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
    pid = tl.program_id(0)
    start_row = pid * ROWS_PER_PROG
    offs = tl.arange(0, D)          # full row offsets
    offs_half = tl.arange(0, HALF_D)

    for i in range(ROWS_PER_PROG):
        row = start_row + i
        # decompose linear row index into (b, h, s)
        b = row // (H * S)
        rem = row % (H * S)
        h = rem // S
        s = rem % S

        # base pointers for this row
        x_base = (x_ptr + b * stride_x_b + h * stride_x_h + s * stride_x_s)
        out_base = (out_ptr + b * stride_out_b + h * stride_out_h + s * stride_out_s)

        # load the entire row (128 contiguous fp16 = 256 bytes)
        # use alignment hint since row start is multiple of 256 bytes
        x = tl.load(tl.multiple_of(x_base, 256) + offs,
                    eviction_policy='evict_first')

        # load cos and sin (64 elements each, aligned to 128 bytes)
        cos_base = cos_ptr + s * stride_cos_s
        sin_base = sin_ptr + s * stride_sin_s
        c = tl.load(tl.multiple_of(cos_base, 128) + offs_half * stride_cos_d,
                    eviction_policy='evict_first')
        s_val = tl.load(tl.multiple_of(sin_base, 128) + offs_half * stride_sin_d,
                        eviction_policy='evict_first')

        # split x into two halves using reshape
        x_2d = tl.reshape(x, (2, HALF_D))
        x1 = x_2d[0, :]
        x2 = x_2d[1, :]

        # compute rotated halves
        out1 = x1 * c - x2 * s_val
        out2 = x1 * s_val + x2 * c

        # combine and store the full row (aligned)
        out_full = tl.cat(out1, out2)
        tl.store(tl.multiple_of(out_base, 256) + offs, out_full,
                 eviction_policy='evict_first')


def triton_run(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float16
    assert cos.dtype == torch.float16 and sin.dtype == torch.float16
    B, H, S, D = x.shape
    assert D == 128
    assert cos.shape == (S, D // 2) and sin.shape == (S, D // 2)

    out = torch.empty_like(x)

    HALF_D = D // 2
    ROWS_PER_PROG = 8
    total_rows = B * H * S
    assert total_rows % ROWS_PER_PROG == 0, "total_rows must be divisible by ROWS_PER_PROG"

    grid = (total_rows // ROWS_PER_PROG,)

    rope_kernel[grid](
        x, cos, sin, out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, S, D,
        HALF_D,
        ROWS_PER_PROG,
        num_warps=4,
        num_stages=2,
    )
    return out