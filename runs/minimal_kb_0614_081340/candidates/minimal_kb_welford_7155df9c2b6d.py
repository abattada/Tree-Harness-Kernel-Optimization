import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel: per‑row mean and population variance via sum / sum-of-squares
# ---------------------------------------------------------------------------
@triton.jit
def welford_kernel(
    x_ptr,                # [rows, cols]
    mean_ptr,             # [rows]
    var_ptr,              # [rows]
    row_stride: tl.int64, # stride of the rows in x (in elements)
    N     : tl.constexpr, # number of columns per row
    BLOCK : tl.constexpr, # block size (must be >= N)
):
    pid = tl.program_id(0)          # one program per row
    row_offset = pid * row_stride   # start of this row in x

    offsets = row_offset + tl.arange(0, BLOCK)
    mask = tl.arange(0, BLOCK) < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # one-pass reduction: sum and sum of squares
    sum_ = tl.sum(x, axis=0)
    sum_sq = tl.sum(x * x, axis=0)

    # compute mean and population variance (numerically stable enough)
    N_f = N.to(tl.float32)
    mean = sum_ / N_f
    var  = (sum_sq / N_f) - mean * mean   # unbiased=False

    tl.store(mean_ptr + pid, mean)
    tl.store(var_ptr  + pid, var)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    x : (8192, 4096)  float32
    returns: (2, 8192) float32  [0] = mean, [1] = population variance
    """
    rows, N = x.shape
    assert rows == 8192 and N == 4096, "Expected (8192, 4096)"

    output = torch.empty(2, rows, device=x.device, dtype=torch.float32)
    mean_ptr = output[0]   # shape [rows]
    var_ptr  = output[1]   # shape [rows]

    # Launch configuration
    BLOCK = 4096
    grid = (rows,)
    welford_kernel[grid](
        x, mean_ptr, var_ptr,
        row_stride=x.stride(0),
        N=N,
        BLOCK=BLOCK,
        num_warps=8,        # good balance for 4096 elements
    )
    return output