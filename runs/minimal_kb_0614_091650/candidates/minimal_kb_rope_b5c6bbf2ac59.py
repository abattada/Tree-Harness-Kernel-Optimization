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
    HALF_D: tl.constexpr, ROWS_PER_PROG: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s_block = tl.program_id(2)

    s_start = pid_s_block * ROWS_PER_PROG

    offs_half = tl.arange(0, HALF_D)
    offs_d = tl.arange(0, D)

    for i in range(ROWS_PER_PROG):
        s = s_start + i
        if s >= S:
            break

        # Base offsets for this row
        base_x = pid_b * stride_x_b + pid_h * stride_x_h + s * stride_x_s
        base_out = pid_b * stride_out_b + pid_h * stride_out_h + s * stride_out_s

        # Load first half x1 (0..63)
        x1 = tl.load(x_ptr + base_x + offs_half * stride_x_d,
                     mask=offs_half < HALF_D,
                     other=0.0,
                     eviction_policy='evict_first')

        # Load second half x2 (64..127)
        x2 = tl.load(x_ptr + base_x + (offs_half + HALF_D) * stride_x_d,
                     mask=offs_half < HALF_D,
                     other=0.0,
                     eviction_policy='evict_first')

        # Load cos and sin for this sequence position
        cos_val = tl.load(cos_ptr + s * stride_cos_s + offs_half * stride_cos_d,
                          mask=offs_half < HALF_D,
                          other=0.0,
                          eviction_policy='evict_first')
        sin_val = tl.load(sin_ptr + s * stride_sin_s + offs_half * stride_sin_d,
                          mask=offs_half < HALF_D,
                          other=0.0,
                          eviction_policy='evict_first')

        # Compute rotated halves
        out1 = x1 * cos_val - x2 * sin_val
        out2 = x1 * sin_val + x2 * cos_val

        # Store results
        tl.store(out_ptr + base_out + offs_half * stride_out_d, out1,
                 mask=offs_half < HALF_D,
                 eviction_policy='evict_first')
        tl.store(out_ptr + base_out + (offs_half + HALF_D) * stride_out_d, out2,
                 mask=offs_half < HALF_D,
                 eviction_policy='evict_first')


def triton_run(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    B, H, S, D = x.shape
    assert D == 128, "RoPE kernel assumes D=128"
    assert cos.shape == (S, D // 2) and sin.shape == (S, D // 2)
    assert x.dtype == torch.float16
    assert cos.dtype == torch.float16 and sin.dtype == torch.float16

    out = torch.empty_like(x)

    HALF_D = D // 2          # 64
    ROWS_PER_PROG = 4       # process 4 rows per program

    num_s_blocks = (S + ROWS_PER_PROG - 1) // ROWS_PER_PROG
    grid = (B, H, num_s_blocks)

    rope_kernel[grid](
        x, cos, sin, out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, S, D,
        HALF_D, ROWS_PER_PROG,
        num_warps=4,
        num_stages=2,
    )
    return out