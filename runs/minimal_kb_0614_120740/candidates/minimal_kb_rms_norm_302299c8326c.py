import torch
import triton
import triton.language as tl

@triton.jit
def rms_norm_persistent_kernel(
    x_ptr, out_ptr,
    M,  # number of rows
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Persistent kernel: one program per SM, each processes a chunk of rows.
    """
    sm_id = tl.program_id(0)
    # Grid-stride loop over rows
    row = sm_id
    while row < M:
        x_row_ptr = x_ptr + row * N
        out_row_ptr = out_ptr + row * N

        offsets = tl.arange(0, BLOCK)
        # Hint contiguous for better vectorization
        offsets = tl.max_contiguous(offsets, BLOCK)

        # Load input row – evict first because it's streamed
        x = tl.load(x_row_ptr + offsets, mask=None, eviction_policy='evict_first')

        # Compute RMS in one pass
        x_sq = x * x
        sum_sq = tl.sum(x_sq, axis=0)
        mean_sq = sum_sq / N
        rstd = tl.rsqrt(mean_sq + eps)

        out = x * rstd
        # Store output – hint evict last for potential reuse
        tl.store(out_row_ptr + offsets, out, mask=None, eviction_policy='evict_last')

        row += M  # actual stride is number of SMs; compute on CPU and pass as stride
        # Actually we need the number of SMs as a constant for stride.
        # We'll compute stride = num_sms on the host and pass as integer argument.
        # So we need another argument for stride.

@triton.jit
def rms_norm_persistent_kernel_v2(
    x_ptr, out_ptr,
    M, stride_sms,  # stride = number of SMs
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    sm_id = tl.program_id(0)
    row = sm_id
    while row < M:
        x_row_ptr = x_ptr + row * N
        out_row_ptr = out_ptr + row * N

        offsets = tl.arange(0, BLOCK)
        offsets = tl.max_contiguous(offsets, BLOCK)

        x = tl.load(x_row_ptr + offsets, mask=None, eviction_policy='evict_first')

        x_sq = x * x
        sum_sq = tl.sum(x_sq, axis=0)
        mean_sq = sum_sq / N
        rstd = tl.rsqrt(mean_sq + eps)

        out = x * rstd
        tl.store(out_row_ptr + offsets, out, mask=None, eviction_policy='evict_last')

        row += stride_sms


def triton_run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous()
    M, N = x.shape
    out = torch.empty_like(x)

    # Query device properties for number of SMs
    device = x.device
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count

    BLOCK = N  # 4096, exactly fills a row
    # Launch one program per SM, each will loop over rows in a grid-stride pattern
    grid = (num_sms,)

    rms_norm_persistent_kernel_v2[grid](
        x, out,
        M, num_sms,
        N=N,
        eps=1e-5,
        BLOCK=BLOCK,
        num_warps=8,
        num_stages=1,
    )
    return out