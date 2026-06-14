import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def geglu_kernel(
    x_ptr,
    out_ptr,
    stride_x,
    stride_out,
    M,              # total number of rows (boundary check)
    rows_per_prog,  # dynamic loop bound to avoid unrolling and register bloat
    BLOCK_SIZE: tl.constexpr,
    N: tl.constexpr,
):
    pid = tl.program_id(0)
    col_offs = tl.arange(0, BLOCK_SIZE)

    # Process multiple rows per program to amortise launch overhead
    for r in range(rows_per_prog):
        row = pid * rows_per_prog + r
        # Boundary check – rows_per_prog may not divide M exactly
        if row < M:
            x_row = x_ptr + row * stride_x
            out_row = out_ptr + row * stride_out

            # Load a (first half) and b (second half) with evict_first for streaming
            a = tl.load(x_row + col_offs, eviction_policy='evict_first')
            b = tl.load(x_row + col_offs + N, eviction_policy='evict_first')

            # GELU with tanh approximation:
            #   GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            a3 = a * a * a
            sqrt2_over_pi = 0.7978845608028654
            c = 0.044715
            inner = sqrt2_over_pi * (a + c * a3)
            tanh_val = libdevice.tanh(inner)
            gelu_a = 0.5 * a * (1.0 + tanh_val)

            # Gating
            out = gelu_a * b

            tl.store(out_row + col_offs, out, eviction_policy='evict_first')


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    x: float32 tensor of shape [M, K]  (K even, here 8192 x 8192)
    returns: float32 tensor of shape [M, K//2]
    """
    M, K = x.shape
    N = K // 2                     # 4096
    out = torch.empty(M, N, device=x.device, dtype=torch.float32)

    BLOCK_SIZE = N                 # covers a full output row → masks unnecessary
    rows_per_prog = 8              # process 8 rows per program
    grid = (triton.cdiv(M, rows_per_prog),)

    geglu_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        M, rows_per_prog,
        BLOCK_SIZE=BLOCK_SIZE,
        N=N,
        num_warps=8,
        num_stages=4,
    )
    return out