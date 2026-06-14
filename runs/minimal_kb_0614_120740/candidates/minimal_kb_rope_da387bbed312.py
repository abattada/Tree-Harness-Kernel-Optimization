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
    offs_half = tl.arange(0, HALF_D)

    # Alignment hints for contiguous dimension (d)
    # The pointer for each load will be wrapped with max_contiguous to help vectorization
    for i in range(ROWS_PER_PROG):
        row = start_row + i
        b = row // (H * S)
        rem = row % (H * S)
        h = rem // S
        s = rem % S

        # Compute base pointers for x and output
        x_base = x_ptr + b * stride_x_b + h * stride_x_h + s * stride_x_s
        out_base = out_ptr + b * stride_out_b + h * stride_out_h + s * stride_out_s

        # Load x halves with alignment hints
        ptr_x1 = x_base + offs_half
        ptr_x1 = tl.max_contiguous(ptr_x1, HALF_D)
        x1 = tl.load(ptr_x1, eviction_policy='evict_first')

        ptr_x2 = x_base + (offs_half + HALF_D)
        ptr_x2 = tl.max_contiguous(ptr_x2, HALF_D)
        x2 = tl.load(ptr_x2, eviction_policy='evict_first')

        # Load cos and sin with alignment hints
        cos_base = cos_ptr + s * stride_cos_s
        ptr_cos = cos_base + offs_half
        ptr_cos = tl.max_contiguous(ptr_cos, HALF_D)
        c = tl.load(ptr_cos, eviction_policy='evict_first')

        sin_base = sin_ptr + s * stride_sin_s
        ptr_sin = sin_base + offs_half
        ptr_sin = tl.max_contiguous(ptr_sin, HALF_D)
        s_val = tl.load(ptr_sin, eviction_policy='evict_first')

        # Compute rotated halves
        out1 = x1 * c - x2 * s_val
        out2 = x1 * s_val + x2 * c

        # Store output with alignment hints
        ptr_out1 = out_base + offs_half
        ptr_out1 = tl.max_contiguous(ptr_out1, HALF_D)
        tl.store(ptr_out1, out1)

        ptr_out2 = out_base + (offs_half + HALF_D)
        ptr_out2 = tl.max_contiguous(ptr_out2, HALF_D)
        tl.store(ptr_out2, out2)


def triton_run(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float16
    assert cos.dtype == torch.float16 and sin.dtype == torch.float16
    B, H, S, D = x.shape
    assert D == 128
    assert cos.shape == (S, D // 2) and sin.shape == (S, D // 2)

    out = torch.empty_like(x)

    HALF_D = D // 2  # 64
    ROWS_PER_PROG = 32  # Increased from 16 to reduce grid size and improve occupancy
    total_rows = B * H * S
    assert total_rows % ROWS_PER_PROG == 0, "grid-stride loop assumes exact division"

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