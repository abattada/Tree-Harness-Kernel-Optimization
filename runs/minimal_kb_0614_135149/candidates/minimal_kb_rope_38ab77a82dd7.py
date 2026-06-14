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
):
    # Persistent grid-stride loop – one program processes many rows
    pid = tl.program_id(0)
    total_rows = B * H * S
    offs_half = tl.arange(0, HALF_D)  # 64 elements, perfect for 2-warps

    for row in range(pid, total_rows, tl.num_programs(0)):
        # Linear row -> (b, h, s)
        b = row // (H * S)
        rem = row % (H * S)
        h = rem // S
        s = rem % S

        base_x = b * stride_x_b + h * stride_x_h + s * stride_x_s
        base_out = b * stride_out_b + h * stride_out_h + s * stride_out_s

        # Load halves and trig values (coalesced, full thread utilisation)
        x1 = tl.load(x_ptr + base_x + offs_half * stride_x_d,
                     eviction_policy='evict_first')
        x2 = tl.load(x_ptr + base_x + (offs_half + HALF_D) * stride_x_d,
                     eviction_policy='evict_first')
        c = tl.load(cos_ptr + s * stride_cos_s + offs_half * stride_cos_d,
                    eviction_policy='evict_first')
        s_val = tl.load(sin_ptr + s * stride_sin_s + offs_half * stride_sin_d,
                        eviction_policy='evict_first')

        # Rotate
        out1 = x1 * c - x2 * s_val
        out2 = x1 * s_val + x2 * c

        # Store
        tl.store(out_ptr + base_out + offs_half * stride_out_d, out1)
        tl.store(out_ptr + base_out + (offs_half + HALF_D) * stride_out_d, out2)


def triton_run(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float16
    assert cos.dtype == torch.float16 and sin.dtype == torch.float16
    B, H, S, D = x.shape
    assert D == 128
    assert cos.shape == (S, D // 2) and sin.shape == (S, D // 2)

    out = torch.empty_like(x)

    HALF_D = D // 2  # 64
    total_rows = B * H * S      # 8 * 32 * 4096 = 1_048_576
    grid_size = min(total_rows, 4096)  # keep SMs busy without excessive launch overhead
    grid = (grid_size,)

    rope_kernel[grid](
        x, cos, sin, out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B=B, H=H, S=S, D=D, HALF_D=HALF_D,
        num_warps=2,      # 64 threads → perfect 1‑1 mapping to the 64‑element loads
        num_stages=3,     # additional pipelining of memory requests
    )
    return out