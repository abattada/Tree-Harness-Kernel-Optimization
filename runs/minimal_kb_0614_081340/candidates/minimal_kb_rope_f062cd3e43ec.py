import torch
import triton
import triton.language as tl

@triton.jit
def rope_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    B, H, S, D,
    stride_x_b, stride_x_h, stride_x_s, stride_x_d,
    stride_cos_s, stride_cos_d,
    stride_sin_s, stride_sin_d,
    stride_out_b, stride_out_h, stride_out_s, stride_out_d,
    BLOCK_D: tl.constexpr, HALF_D: tl.constexpr, ROWS_PER_PROG: tl.constexpr,
):
    total_rows = B * H * S
    pid = tl.program_id(0)
    start_row = pid * ROWS_PER_PROG

    for i in range(ROWS_PER_PROG):
        row = start_row + i
        if row >= total_rows:
            break

        # Decompose the flat row index into (b, h, s)
        b = row // (H * S)
        rem = row % (H * S)
        h = rem // S
        s = rem % S

        # Base pointers for x and out for this row
        x_base = x_ptr + b * stride_x_b + h * stride_x_h + s * stride_x_s
        out_base = out_ptr + b * stride_out_b + h * stride_out_h + s * stride_out_s

        offs_half = tl.arange(0, HALF_D)

        # Load halves of x (contiguous, no mask needed)
        x1 = tl.load(x_base + offs_half)
        x2 = tl.load(x_base + offs_half + HALF_D)

        # Load cos and sin for this sequence position
        cos_ptr_s = cos_ptr + s * stride_cos_s
        sin_ptr_s = sin_ptr + s * stride_sin_s
        cos_vals = tl.load(cos_ptr_s + offs_half)
        sin_vals = tl.load(sin_ptr_s + offs_half)

        # Rotate halves
        y1 = x1 * cos_vals - x2 * sin_vals
        y2 = x1 * sin_vals + x2 * cos_vals

        # Store output halves
        tl.store(out_base + offs_half, y1)
        tl.store(out_base + offs_half + HALF_D, y2)


def triton_run(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    B, H, S, D = x.shape
    assert D == 128, "RoPE kernel expects D=128"
    assert cos.shape == (S, D // 2), "cos shape mismatch"
    assert sin.shape == (S, D // 2), "sin shape mismatch"

    out = torch.empty_like(x)

    total_rows = B * H * S
    # Process 8 rows per program to reduce grid size and improve occupancy
    ROWS_PER_PROG = 8
    grid = (triton.cdiv(total_rows, ROWS_PER_PROG),)

    rope_kernel[grid](
        x, cos, sin, out,
        B, H, S, D,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_D=128, HALF_D=64, ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=4,
        num_stages=2,
    )
    return out