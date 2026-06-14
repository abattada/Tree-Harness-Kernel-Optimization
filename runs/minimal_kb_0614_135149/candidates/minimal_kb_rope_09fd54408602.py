import torch
import triton
import triton.language as tl


@triton.jit
def rope_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    stride_x_b,
    stride_x_h,
    stride_x_s,
    stride_x_d,
    stride_cos_s,
    stride_cos_d,
    stride_sin_s,
    stride_sin_d,
    stride_out_b,
    stride_out_h,
    stride_out_s,
    stride_out_d,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    HALF_D: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    """RoPE kernel: each program processes ROWS_PER_PROG rows.
       Optimized for num_warps=2: exactly 64 threads cover the 64-elem half dimension.
    """
    pid = tl.program_id(0)
    start_row = pid * ROWS_PER_PROG
    offs_half = tl.arange(0, HALF_D)  # 0 .. 63

    for i in range(ROWS_PER_PROG):
        row = start_row + i
        # Decompose linear row index into (batch, head, seq)
        b = row // (H * S)
        rem = row % (H * S)
        h = rem // S
        s = rem % S

        # Base addresses for x and output at this row
        base_x = x_ptr + b * stride_x_b + h * stride_x_h + s * stride_x_s
        base_out = out_ptr + b * stride_out_b + h * stride_out_h + s * stride_out_s

        # Load x halves (contiguous, stride == 1 element)
        x1 = tl.load(base_x + offs_half * stride_x_d, eviction_policy='evict_first')
        x2 = tl.load(base_x + (offs_half + HALF_D) * stride_x_d, eviction_policy='evict_first')

        # Load cos and sin for this sequence position
        c = tl.load(cos_ptr + s * stride_cos_s + offs_half * stride_cos_d, eviction_policy='evict_first')
        s_vals = tl.load(sin_ptr + s * stride_sin_s + offs_half * stride_sin_d, eviction_policy='evict_first')

        # RoPE arithmetic
        out1 = x1 * c - x2 * s_vals
        out2 = x1 * s_vals + x2 * c

        # Store results
        tl.store(base_out + offs_half * stride_out_d, out1)
        tl.store(base_out + (offs_half + HALF_D) * stride_out_d, out2)


def triton_run(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float16
    assert cos.dtype == torch.float16 and sin.dtype == torch.float16
    B, H, S, D = x.shape
    assert D == 128, "expected last dim 128"
    assert cos.shape == (S, D // 2) and sin.shape == (S, D // 2), "cos/sin shape mismatch"

    out = torch.empty_like(x)

    HALF_D = D // 2  # 64
    # Process many rows per program to reduce launch overhead, keep grid moderate.
    # total_rows = 8*32*4096 = 1,048,576; divisible by 64 -> grid=16384
    ROWS_PER_PROG = 64
    total_rows = B * H * S
    assert total_rows % ROWS_PER_PROG == 0, "total rows must be a multiple of ROWS_PER_PROG"

    grid = (total_rows // ROWS_PER_PROG,)

    rope_kernel[grid](
        x, cos, sin, out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B=B, H=H, S=S, D=D,
        HALF_D=HALF_D,
        ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=2,    # exactly 64 threads – full utilisation for the 64-element half
        num_stages=2,   # keep pipelining for memory loads
    )
    return out