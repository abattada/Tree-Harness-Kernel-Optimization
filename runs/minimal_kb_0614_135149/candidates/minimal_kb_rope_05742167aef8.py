import torch
import triton
import triton.language as tl


@triton.jit
def rope_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    stride_x_b, stride_x_h, stride_x_s,
    stride_cos_s, stride_cos_d,
    stride_sin_s, stride_sin_d,
    stride_out_b, stride_out_h, stride_out_s,
    B, H, S, D: tl.constexpr, HALF_D: tl.constexpr,
    S_PER_PROG: tl.constexpr,
):
    """
    Reorder loops so that for each sequence position `s` we load cos/sin once
    and then iterate over all (batch, head) rows.  This removes the redundant
    reloading of cos/sin that consumed half of the original memory traffic.
    """
    pid = tl.program_id(0)
    offs_half = tl.arange(0, HALF_D)  # 0 .. 63

    for s_off in range(S_PER_PROG):
        s = pid * S_PER_PROG + s_off

        # cos/sin for this s – loaded once and reused for all (b, h)
        cos_vals = tl.load(cos_ptr + s * stride_cos_s + offs_half * stride_cos_d,
                           eviction_policy='evict_last')
        sin_vals = tl.load(sin_ptr + s * stride_sin_s + offs_half * stride_sin_d,
                           eviction_policy='evict_last')

        for b in range(B):
            for h in range(H):
                base_x = b * stride_x_b + h * stride_x_h + s * stride_x_s
                base_out = b * stride_out_b + h * stride_out_h + s * stride_out_s

                # Load x halves – streaming access, evict first
                x1 = tl.load(x_ptr + base_x + offs_half, eviction_policy='evict_first')
                x2 = tl.load(x_ptr + base_x + offs_half + HALF_D, eviction_policy='evict_first')

                out1 = x1 * cos_vals - x2 * sin_vals
                out2 = x1 * sin_vals + x2 * cos_vals

                tl.store(out_ptr + base_out + offs_half, out1)
                tl.store(out_ptr + base_out + offs_half + HALF_D, out2)


def triton_run(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float16
    assert cos.dtype == torch.float16 and sin.dtype == torch.float16
    B, H, S, D = x.shape
    assert D == 128
    assert cos.shape == (S, D // 2)
    assert sin.shape == (S, D // 2)

    out = torch.empty_like(x)
    HALF_D = D // 2

    S_PER_PROG = 8
    assert S % S_PER_PROG == 0

    grid = (S // S_PER_PROG,)

    rope_kernel[grid](
        x, cos, sin, out,
        x.stride(0), x.stride(1), x.stride(2),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        B, H, S, D, HALF_D,
        S_PER_PROG,
        num_warps=8,
        num_stages=3,
    )
    return out