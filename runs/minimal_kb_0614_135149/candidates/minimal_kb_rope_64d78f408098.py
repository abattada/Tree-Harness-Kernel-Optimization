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
    stride_cos_s,
    stride_sin_s,
    stride_out_b,
    stride_out_h,
    stride_out_s,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    HALF_D: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    """
    RoPE kernel: each program processes ROWS_PER_PROG consecutive sequence positions
    for a single (batch, head) pair.
    Grid: (B * H, S // ROWS_PER_PROG)
    """
    pid_bh = tl.program_id(0)   # 0 … B*H - 1
    pid_s  = tl.program_id(1)   # 0 … S//ROWS_PER_PROG - 1

    b   = pid_bh // H
    h   = pid_bh % H
    s0  = pid_s * ROWS_PER_PROG

    offs_half = tl.arange(0, HALF_D)

    for soff in range(ROWS_PER_PROG):
        s = s0 + soff

        # base addresses for this row
        base_x   = b * stride_x_b + h * stride_x_h + s * stride_x_s
        base_out = b * stride_out_b + h * stride_out_h + s * stride_out_s

        # x1, x2 are contiguous in the last dim (stride == 1, enforced in launch)
        x1 = tl.load(x_ptr + base_x + offs_half, eviction_policy='evict_first')
        x2 = tl.load(x_ptr + base_x + offs_half + HALF_D, eviction_policy='evict_first')

        # cos and sin are also contiguous in the half-dim
        c  = tl.load(cos_ptr + s * stride_cos_s + offs_half, eviction_policy='evict_first')
        sv = tl.load(sin_ptr + s * stride_sin_s + offs_half, eviction_policy='evict_first')

        out1 = x1 * c  - x2 * sv
        out2 = x1 * sv + x2 * c

        tl.store(out_ptr + base_out + offs_half, out1)
        tl.store(out_ptr + base_out + offs_half + HALF_D, out2)


def triton_run(x, cos, sin):
    """
    RoPE: rotate the last dimension of x using cos/sin tables.
    Expects x (f16, 8×32×4096×128), cos & sin (f16, 4096×64).
    """
    assert x.dtype == torch.float16
    assert cos.dtype == torch.float16 and sin.dtype == torch.float16

    B, H, S, D = x.shape
    assert D == 128
    assert cos.shape == (S, D // 2) and sin.shape == (S, D // 2)

    # Require contiguous last dimension – usual for this operator
    assert x.stride(3) == 1
    assert cos.stride(1) == 1
    assert sin.stride(1) == 1

    out = torch.empty_like(x)

    HALF_D = D // 2          # 64
    ROWS_PER_PROG = 8        # processes 8 rows per program
    assert S % ROWS_PER_PROG == 0, "S must be divisible by ROWS_PER_PROG"

    # 2‑D grid: (batch*head, sequence chunks)
    grid = (B * H, S // ROWS_PER_PROG)

    rope_kernel[grid](
        x, cos, sin, out,
        x.stride(0), x.stride(1), x.stride(2),
        cos.stride(0),
        sin.stride(0),
        out.stride(0), out.stride(1), out.stride(2),
        B=B, H=H, S=S,
        HALF_D=HALF_D,
        ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=4,
        num_stages=2,
    )
    return out