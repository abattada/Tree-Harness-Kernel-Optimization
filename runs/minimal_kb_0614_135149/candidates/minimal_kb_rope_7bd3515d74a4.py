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
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    HALF_D: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    TOTAL_ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)  # equals NUM_PROGRAMS
    offs_half = tl.arange(0, HALF_D)  # 64 elements, perfectly matches 2 warps

    # Grid-stride loop: each program processes multiple tiles of ROWS_PER_PROG rows
    step = num_progs * ROWS_PER_PROG  # compile-time known, constexpr
    for start_row in range(pid * ROWS_PER_PROG, TOTAL_ROWS, step):
        for i in range(ROWS_PER_PROG):
            row = start_row + i
            # Decompose linear row index into (b, h, s)
            b = row // (H * S)
            rem = row % (H * S)
            h = rem // S
            s = rem % S

            # Base offsets for x and output at this row
            x_base = x_ptr + b * stride_x_b + h * stride_x_h + s * stride_x_s
            out_base = out_ptr + b * stride_out_b + h * stride_out_h + s * stride_out_s

            # Load x halves (contiguous, no mask needed)
            x1 = tl.load(x_base + offs_half * stride_x_d, eviction_policy='evict_first')
            x2 = tl.load(x_base + (offs_half + HALF_D) * stride_x_d, eviction_policy='evict_first')

            # Load cos and sin for this sequence position
            cos_base = cos_ptr + s * stride_cos_s
            sin_base = sin_ptr + s * stride_sin_s
            c = tl.load(cos_base + offs_half * stride_cos_d, eviction_policy='evict_first')
            s_val = tl.load(sin_base + offs_half * stride_sin_d, eviction_policy='evict_first')

            # Compute rotated halves
            out1 = x1 * c - x2 * s_val
            out2 = x1 * s_val + x2 * c

            # Store output
            tl.store(out_base + offs_half * stride_out_d, out1, eviction_policy='evict_first')
            tl.store(out_base + (offs_half + HALF_D) * stride_out_d, out2, eviction_policy='evict_first')


def triton_run(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float16
    assert cos.dtype == torch.float16 and sin.dtype == torch.float16
    B, H, S, D = x.shape
    assert D == 128
    assert cos.shape == (S, D // 2) and sin.shape == (S, D // 2)

    out = torch.empty_like(x)

    HALF_D = D // 2          # 64
    ROWS_PER_PROG = 8
    NUM_PROGRAMS = 2048      # persistent kernel grid size
    TOTAL_ROWS = B * H * S   # 8*32*4096 = 1048576

    # Ensure exact divisibility so no boundary masks are needed
    assert TOTAL_ROWS % (ROWS_PER_PROG * NUM_PROGRAMS) == 0, \
        "TOTAL_ROWS must be divisible by ROWS_PER_PROG * NUM_PROGRAMS"

    grid = (NUM_PROGRAMS,)

    rope_kernel[grid](
        x, cos, sin, out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, S, D,
        HALF_D,
        ROWS_PER_PROG,
        NUM_PROGRAMS,
        TOTAL_ROWS,
        num_warps=4,
        num_stages=3,
    )
    return out