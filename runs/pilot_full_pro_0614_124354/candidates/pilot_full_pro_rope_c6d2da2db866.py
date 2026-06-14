import torch
import triton
import triton.language as tl
import math

@triton.jit
def rope_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    num_rows: tl.int,
    T: tl.int,
    D: tl.constexpr,
    D2: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Vectorized RoPE kernel processing ROWS_PER_PROG rows per program."""
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROG

    # Shared memory to exchange x elements between threads
    smem_x = tl.static_shared(shape=(D,), dtype=tl.float16)

    for r in range(ROWS_PER_PROG):
        row_idx = row_start + r
        valid = row_idx < num_rows
        safe_row_idx = tl.where(valid, row_idx, 0)
        t = safe_row_idx % T

        # Row base pointers (elements, not bytes)
        x_row = x_ptr + safe_row_idx * D
        cos_row = cos_ptr + t * D2
        sin_row = sin_ptr + t * D2
        out_row = out_ptr + safe_row_idx * D

        # Load full row into shared memory
        offs = tl.arange(0, BLOCK)  # BLOCK == D == 128
        x_vals = tl.load(x_row + offs)
        tl.store(smem_x + offs, x_vals)
        tl.debug_barrier()  # make row visible to all threads

        # Determine first‑half vs second‑half indices
        cond = offs < D2
        partner_offs = tl.where(cond, offs + D2, offs - D2)
        partner_vals = tl.load(smem_x + partner_offs)

        # Load rotation coefficients (each thread loads its own)
        d_cos = tl.where(cond, offs, offs - D2)
        cos_vals = tl.load(cos_row + d_cos)
        sin_vals = tl.load(sin_row + d_cos)

        # RoPE transform
        out_vals = tl.where(
            cond,
            x_vals * cos_vals - partner_vals * sin_vals,
            partner_vals * sin_vals + x_vals * cos_vals,
        )

        # Store result (masked for the very last chunk)
        tl.store(out_row + offs, out_vals, mask=valid)

        # Block‑wide barrier before reusing shared memory
        tl.debug_barrier()


def triton_run(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Rotary position embedding: out = cat(x1*c - x2*s, x1*s + x2*c)
    where x1,x2 = split(x, 64, dim=-1).
    """
    assert x.dtype == torch.float16, "x must be float16"
    assert cos.dtype == torch.float16, "cos must be float16"
    assert sin.dtype == torch.float16, "sin must be float16"

    B, H, T, D = x.shape
    D2 = cos.shape[-1]   # 64
    assert D == 2 * D2, f"Last dim of x must be 2 * D2, got {D} vs {2*D2}"
    assert cos.shape[0] == T and cos.shape[1] == D2
    assert sin.shape[0] == T and sin.shape[1] == D2

    num_rows = B * H * T
    out = torch.empty_like(x)

    # Tuning knobs (to be optimized in follow‑up rounds)
    ROWS_PER_PROG = 8
    BLOCK = D  # 128 – one thread per output element

    grid = (triton.cdiv(num_rows, ROWS_PER_PROG),)

    rope_kernel[grid](
        x.data_ptr(),
        cos.data_ptr(),
        sin.data_ptr(),
        out.data_ptr(),
        num_rows,
        T,
        D=128,
        D2=64,
        ROWS_PER_PROG=ROWS_PER_PROG,
        BLOCK=BLOCK,
        num_warps=4,      # 4 warps × 32 threads = 128 threads
        num_stages=2,
    )

    return out