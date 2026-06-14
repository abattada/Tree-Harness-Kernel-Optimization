import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def rope_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    stride_x_b, stride_x_h, stride_x_s, stride_x_d,
    stride_cos_s, stride_cos_d,
    stride_sin_s, stride_sin_d,
    stride_out_b, stride_out_h, stride_out_s, stride_out_d,
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    HALF_D: tl.constexpr,  # = D // 2 = 64
    BLOCK_HALF: tl.constexpr,  # = 64
    BLOCK_S: tl.constexpr,    # number of sequence positions per program
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s_blk = tl.program_id(2)

    s_start = pid_s_blk * BLOCK_S

    # Precompute offsets for the half dimension (0..63)
    offsets = tl.arange(0, BLOCK_HALF)

    # Loop over the sequence positions in this block
    for s_offset in range(BLOCK_S):
        s = s_start + s_offset
        if s >= S:
            break

        # Base offsets for x and out at (pid_b, pid_h, s)
        base_x = pid_b * stride_x_b + pid_h * stride_x_h + s * stride_x_s
        base_out = pid_b * stride_out_b + pid_h * stride_out_h + s * stride_out_s

        # Load first half x1 (0..63)
        x1_ptrs = x_ptr + base_x + offsets * stride_x_d
        x1 = tl.load(x1_ptrs, mask=offsets < HALF_D, eviction_policy='evict_first')

        # Load second half x2 (64..127)
        x2_ptrs = x_ptr + base_x + (offsets + HALF_D) * stride_x_d
        x2 = tl.load(x2_ptrs, mask=offsets < HALF_D, eviction_policy='evict_first')

        # Load cos and sin for this sequence position
        cos_ptrs = cos_ptr + s * stride_cos_s + offsets * stride_cos_d
        sin_ptrs = sin_ptr + s * stride_sin_s + offsets * stride_sin_d
        c = tl.load(cos_ptrs, mask=offsets < HALF_D, eviction_policy='evict_last')
        s_val = tl.load(sin_ptrs, mask=offsets < HALF_D, eviction_policy='evict_last')

        # Compute rotated halves
        out1 = x1 * c - x2 * s_val
        out2 = x1 * s_val + x2 * c

        # Store out1 at first half
        out1_ptrs = out_ptr + base_out + offsets * stride_out_d
        tl.store(out1_ptrs, out1, mask=offsets < HALF_D, eviction_policy='evict_first')

        # Store out2 at second half
        out2_ptrs = out_ptr + base_out + (offsets + HALF_D) * stride_out_d
        tl.store(out2_ptrs, out2, mask=offsets < HALF_D, eviction_policy='evict_first')


def triton_run(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float16
    assert cos.dtype == torch.float16 and sin.dtype == torch.float16
    B, H, S, D = x.shape
    assert D == 128
    assert cos.shape == (S, D // 2) and sin.shape == (S, D // 2)

    out = torch.empty_like(x)

    HALF_D = D // 2  # 64
    BLOCK_HALF = 64
    BLOCK_S = 4  # process 4 sequence positions per program

    grid = (B, H, (S + BLOCK_S - 1) // BLOCK_S)

    rope_kernel[grid](
        x, cos, sin, out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, S, D,
        HALF_D, BLOCK_HALF, BLOCK_S,
    )

    return out