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
    total_rows = B * H * S
    pid = tl.program_id(0)
    start_row = pid * ROWS_PER_PROG

    offs_half = tl.arange(0, HALF_D)

    for i in tl.static_range(ROWS_PER_PROG):
        row = start_row + i
        if row >= total_rows:
            break

        # Decompose linear row index into (batch, head, seq)
        b = row // (H * S)
        rem = row % (H * S)
        h = rem // S
        s = rem % S

        # Base pointers for this row
        base_x = x_ptr + b * stride_x_b + h * stride_x_h + s * stride_x_s
        base_out = out_ptr + b * stride_out_b + h * stride_out_h + s * stride_out_s
        cos_base = cos_ptr + s * stride_cos_s
        sin_base = sin_ptr + s * stride_sin_s

        # Load the two halves of x, and the corresponding cos/sin slices
        x1 = tl.load(base_x + offs_half * stride_x_d, eviction_policy='evict_first')
        x2 = tl.load(base_x + (offs_half + HALF_D) * stride_x_d,
                     eviction_policy='evict_first')
        c = tl.load(cos_base + offs_half * stride_cos_d, eviction_policy='evict_first')
        s_val = tl.load(sin_base + offs_half * stride_sin_d,
                        eviction_policy='evict_first')

        # Rotate half
        out1 = x1 * c - x2 * s_val
        out2 = x1 * s_val + x2 * c

        # Store the two halves contiguously
        tl.store(base_out + offs_half * stride_out_d, out1)
        tl.store(base_out + (offs_half + HALF_D) * stride_out_d, out2)


def triton_run(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    B, H, S, D = x.shape
    HALF_D = D // 2
    assert cos.shape == (S, HALF_D) and sin.shape == (S, HALF_D)

    out = torch.empty_like(x)

    ROWS_PER_PROG = 8
    total_rows = B * H * S
    grid = (triton.cdiv(total_rows, ROWS_PER_PROG),)

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