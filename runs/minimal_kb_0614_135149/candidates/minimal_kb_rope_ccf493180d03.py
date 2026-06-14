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
    W: tl.constexpr,
    VEC: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Each block processes ROWS_PER_PROG rows.
    Within a block, W warps each process one row per iteration
    using vectorized loads (VEC elements per thread) for the half-dimension.
    """
    pid = tl.program_id(0)
    start_row = pid * ROWS_PER_PROG

    tid = tl.arange(0, BLOCK_SIZE)
    warp_id = tid // 32
    lane_id = tid % 32

    # Loop over rows in chunks of W
    for i in range(0, ROWS_PER_PROG, W):
        # Row index for each warp
        row = start_row + i + warp_id  # shape [BLOCK_SIZE], same across a warp

        # Decompose into batch, head, sequence
        b = row // (H * S)
        rem = row % (H * S)
        h = rem // S
        s = rem % S

        # Base pointers for x and output at this row
        base_x = x_ptr + b * stride_x_b + h * stride_x_h + s * stride_x_s
        base_out = out_ptr + b * stride_out_b + h * stride_out_h + s * stride_out_s

        # Vectorized offsets for the half-dimension: VEC elements per thread
        offs = (lane_id * VEC)[:, None] + tl.arange(0, VEC)  # [BLOCK_SIZE, VEC]

        # Load x halves (stride_x_d == 1, contiguous)
        x1 = tl.load(base_x + offs,
                     eviction_policy='evict_first')
        x2 = tl.load(base_x + (offs + HALF_D) * stride_x_d,
                     eviction_policy='evict_first')

        # Load cos and sin for the sequence position (contiguous along last dim)
        cos_vals = tl.load(cos_ptr + s * stride_cos_s + offs * stride_cos_d,
                           eviction_policy='evict_first')
        sin_vals = tl.load(sin_ptr + s * stride_sin_s + offs * stride_sin_d,
                           eviction_policy='evict_first')

        # RoPE arithmetic
        out1 = x1 * cos_vals - x2 * sin_vals
        out2 = x1 * sin_vals + x2 * cos_vals

        # Store results
        tl.store(base_out + offs, out1)
        tl.store(base_out + (offs + HALF_D) * stride_out_d, out2)


def triton_run(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float16
    assert cos.dtype == torch.float16 and sin.dtype == torch.float16
    B, H, S, D = x.shape
    assert D == 128
    assert cos.shape == (S, D // 2) and sin.shape == (S, D // 2)

    out = torch.empty_like(x)

    HALF_D = D // 2  # 64
    ROWS_PER_PROG = 64
    W = 8          # number of warps per block
    VEC = 2        # vector width per thread
    BLOCK_SIZE = W * 32  # 256

    total_rows = B * H * S
    assert total_rows % ROWS_PER_PROG == 0, "total rows must be multiple of ROWS_PER_PROG"

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
        W=W,
        VEC=VEC,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=W,
        num_stages=3,
    )
    return out