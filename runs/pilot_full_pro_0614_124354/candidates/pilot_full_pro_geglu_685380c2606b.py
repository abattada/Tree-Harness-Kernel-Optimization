import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def geglu_kernel(
    x_ptr,
    y_ptr,
    M: tl.constexpr,
    out_dim: tl.constexpr,
    in_dim: tl.constexpr,
    VEC: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    tid = tl.arange(0, VEC)                     # thread id as a vector of offsets inside the vector chunk
    # Each thread handles VEC contiguous columns.
    # Number of threads per block = out_dim // VEC, which is 128 for VEC=32 and out_dim=4096.

    # Outer loop over rows assigned to this program
    for r in range(ROWS_PER_PROG):
        row = pid * ROWS_PER_PROG + r
        if row < M:
            col_start = tl.arange(0, VEC) * 0 + tid * VEC  # tid * VEC + arange(VEC) effectively: tid * VEC + tl.arange(VEC)
            # Actually simpler: just compute column offsets as tid * VEC + tl.arange(0, VEC)
            col_offs = tid[:, None] * VEC + tl.arange(0, VEC)[None, :]  # shape (1, VEC) per thread, but we want vector per thread.
            # In Triton, tid is a scalar, so we use tid * VEC + tl.arange(0, VEC)
            col_offs = tid * VEC + tl.arange(0, VEC)   # shape (VEC,)

            # a and b load addresses (row-major)
            a_offs = row * in_dim + col_offs
            b_offs = row * in_dim + col_offs + out_dim

            # Vectorized loads (no mask needed because out_dim is a multiple of VEC and col_offs < out_dim)
            a_vec = tl.load(x_ptr + a_offs)
            b_vec = tl.load(x_ptr + b_offs)

            # GELU tanh approximation
            sqrt2pi = 0.7978845608028654   # sqrt(2/pi)
            coeff = 0.044715
            a3 = a_vec * a_vec * a_vec
            inner = sqrt2pi * (a_vec + coeff * a3)
            gelu_vec = 0.5 * a_vec * (1.0 + libdevice.tanh(inner))

            out_vec = gelu_vec * b_vec       # elementwise GEGLU

            # Store result
            y_offs = row * out_dim + col_offs
            tl.store(y_ptr + y_offs, out_vec)


def triton_run(x: torch.Tensor) -> torch.Tensor:
    """
    x: f32[8192, 8192]
    returns: f32[8192, 4096]
    """
    assert x.is_cuda and x.dtype == torch.float32
    assert x.shape == (8192, 8192)

    M = 8192
    in_dim = 8192
    out_dim = 4096

    VEC = 32                        # vector width in floats -> 128 B per access
    ROWS_PER_PROG = 4               # tiles several rows per program to reduce launch count
    block_size = out_dim // VEC     # threads per block = 128

    out = torch.empty(M, out_dim, dtype=torch.float32, device=x.device)

    grid = (triton.cdiv(M, ROWS_PER_PROG),)

    geglu_kernel[grid](
        x, out,
        M, out_dim, in_dim,
        VEC, ROWS_PER_PROG,
        num_warps=4, num_stages=3,  # 128 threads -> 4 warps; 3 stages for pipelining
    )
    return out