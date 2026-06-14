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
    """RoPE kernel: each program processes ROWS_PER_PROG consecutive rows.
       Tuned for 2 warps (64 threads) to exactly cover the 64-element half.
       Increased ROWS_PER_PROG reduces launch overhead and improves latency hiding.
    """
    pid = tl.program_id(0)
    start_row = pid * ROWS_PER_PROG
    offs_half = tl.arange(0, HALF_D)  # 0..63

    for i in tl.static_range(ROWS_PER_PROG):
        row = start_row + i
        # Decompose linear row -> (b, h, s)
        b = row // (H * S)
        rem = row % (H * S)
        h = rem // S
        s = rem % S

        # Base addresses for x and output for this row
        x_base = x_ptr + b * stride_x_b + h * stride_x_h + s * stride_x_s
        out_base = out_ptr + b * stride_out_b + h * stride_out_h + s * stride_out_s

        # Load first and second halves of x (contiguous, 128 bytes each -> full cache line)
        x1 = tl.load(x_base + offs_half, eviction_policy="evict_first")
        x2 = tl.load(x_base + offs_half + HALF_D, eviction_policy="evict_first")

        # Load cos / sin for this sequence position
        c = tl.load(cos_ptr + s * stride_cos_s + offs_half * stride_cos_d,
                    eviction_policy="evict_first")
        s_vals = tl.load(sin_ptr + s * stride_sin_s + offs_half * stride_sin_d,
                         eviction_policy="evict_first")

        # RoPE arithmetic
        out1 = x1 * c - x2 * s_vals
        out2 = x1 * s_vals + x2 * c

        # Store the two halves
        tl.store(out_base + offs_half, out1)
        tl.store(out_base + offs_half + HALF_D, out2)


def triton_run(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    B, H, S, D = x.shape
    assert D == 128, "expected last dimension 128"
    assert cos.shape == (S, D // 2) and sin.shape == (S, D // 2)

    out = torch.empty_like(x)

    HALF_D = D // 2  # 64
    # Process many rows per program to keep grid moderate and hide latency.
    # total_rows = 8*32*4096 = 1,048,576  ->  grid = 4096 programs (each 256 rows)
    ROWS_PER_PROG = 256
    total_rows = B * H * S
    assert total_rows % ROWS_PER_PROG == 0

    grid = (total_rows // ROWS_PER_PROG,)

    rope_kernel[grid](
        x,
        cos,
        sin,
        out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        cos.stride(0),
        cos.stride(1),
        sin.stride(0),
        sin.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        B=B,
        H=H,
        S=S,
        D=D,
        HALF_D=HALF_D,
        ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=2,   # exactly 64 threads – perfect for 64-element halves
        num_stages=4,  # deeper pipelining to overlap loads across rows
    )
    return out