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
    """RoPE kernel using tl.make_block_ptr for contiguous vectorised loads/stores."""
    pid = tl.program_id(0)
    start_row = pid * ROWS_PER_PROG

    for i in range(ROWS_PER_PROG):
        row = start_row + i
        b = row // (H * S)
        rem = row % (H * S)
        h = rem // S
        s = rem % S

        x_base = b * stride_x_b + h * stride_x_h + s * stride_x_s
        out_base = b * stride_out_b + h * stride_out_h + s * stride_out_s

        # First half x1 (contiguous, offset 0)
        x1_block = tl.make_block_ptr(
            base=x_ptr + x_base,
            shape=(D,),
            strides=(stride_x_d,),
            offsets=(0,),
            block_shape=(HALF_D,),
            order=(0,),
        )
        x1 = tl.load(x1_block, eviction_policy="evict_first")

        # Second half x2 (contiguous, offset HALF_D)
        x2_block = tl.make_block_ptr(
            base=x_ptr + x_base,
            shape=(D,),
            strides=(stride_x_d,),
            offsets=(HALF_D,),
            block_shape=(HALF_D,),
            order=(0,),
        )
        x2 = tl.load(x2_block, eviction_policy="evict_first")

        # cos for sequence position s
        cos_block = tl.make_block_ptr(
            base=cos_ptr + s * stride_cos_s,
            shape=(HALF_D,),
            strides=(stride_cos_d,),
            offsets=(0,),
            block_shape=(HALF_D,),
            order=(0,),
        )
        c = tl.load(cos_block, eviction_policy="evict_first")

        # sin for sequence position s
        sin_block = tl.make_block_ptr(
            base=sin_ptr + s * stride_sin_s,
            shape=(HALF_D,),
            strides=(stride_sin_d,),
            offsets=(0,),
            block_shape=(HALF_D,),
            order=(0,),
        )
        s_vals = tl.load(sin_block, eviction_policy="evict_first")

        # Rotate halves
        out1 = x1 * c - x2 * s_vals
        out2 = x1 * s_vals + x2 * c

        # Store first half
        out1_block = tl.make_block_ptr(
            base=out_ptr + out_base,
            shape=(D,),
            strides=(stride_out_d,),
            offsets=(0,),
            block_shape=(HALF_D,),
            order=(0,),
        )
        tl.store(out1_block, out1)

        # Store second half
        out2_block = tl.make_block_ptr(
            base=out_ptr + out_base,
            shape=(D,),
            strides=(stride_out_d,),
            offsets=(HALF_D,),
            block_shape=(HALF_D,),
            order=(0,),
        )
        tl.store(out2_block, out2)


def triton_run(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Run RoPE: rotate half the last dimension of x using cos/sin tables."""
    assert x.dtype == torch.float16
    assert cos.dtype == torch.float16 and sin.dtype == torch.float16
    B, H, S, D = x.shape
    assert D == 128, "expected last dim 128"
    assert cos.shape == (S, D // 2) and sin.shape == (S, D // 2)

    out = torch.empty_like(x)

    HALF_D = D // 2  # 64
    ROWS_PER_PROG = 8   # keeps launch overhead low while keeping the grid large enough
    total_rows = B * H * S
    assert total_rows % ROWS_PER_PROG == 0, "total rows must divide ROWS_PER_PROG exactly"

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
        num_warps=4,
        num_stages=2,
    )
    return out