import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    x_ptr, out_ptr,
    M,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Persistent kernel: each program loops over multiple rows
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    offsets = tl.arange(0, BLOCK)

    for row in range(pid, M, num_progs):
        x_row = tl.load(x_ptr + row * N + offsets)
        x_sq = x_row * x_row
        sum_sq = tl.sum(x_sq, axis=0)
        mean_sq = sum_sq / N
        rstd = tl.rsqrt(mean_sq + eps)
        out_row = x_row * rstd
        tl.store(out_ptr + row * N + offsets, out_row)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous()
    M, N = x.shape
    out = torch.empty_like(x)

    BLOCK = N  # 4096 elements per row, exactly one tile per row
    # Launch enough programs to keep SMs busy while reducing launch overhead
    num_progs = min(M, 512)  # 512 is ~4x the number of SMs on RTX 5090
    grid = (num_progs,)

    rms_norm_kernel[grid](
        x, out,
        M,
        N=N,
        eps=1e-5,
        BLOCK=BLOCK,
        num_warps=8,   # 256 threads/block – good balance for this memory-bound kernel
        num_stages=2,  # enables pipelining of loads/stores across loop iterations
    )
    return out