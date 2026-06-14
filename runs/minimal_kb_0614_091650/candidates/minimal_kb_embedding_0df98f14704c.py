import torch
import triton
import triton.language as tl

@triton.jit
def embedding_kernel(
    weight_ptr, idx_ptr, out_ptr,
    N, D,
    stride_w0, stride_w1,
    stride_o0, stride_o1,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    # grid‑stride loop over rows
    row_idx = pid
    while row_idx < N:
        # load index (int64)
        idx = tl.load(idx_ptr + row_idx)

        # base pointer for the selected weight row
        w_row_base = weight_ptr + idx * stride_w0

        # offsets along the embedding dimension (contiguous, no mask)
        offsets = tl.arange(0, BLOCK_D)
        w = tl.load(w_row_base + offsets * stride_w1)

        # store output row
        o_row_base = out_ptr + row_idx * stride_o0
        tl.store(o_row_base + offsets * stride_o1, w)

        # advance to next row handled by this program
        row_idx += tl.num_programs(0)


def triton_run(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    assert weight.dtype == torch.float32
    assert idx.dtype == torch.int64
    assert weight.is_contiguous()

    vocab, D = weight.shape
    N = idx.shape[0]
    # task guarantees D == 1024, but keep it flexible
    assert D == 1024, f"Expected D=1024, got {D}"

    out = torch.empty(N, D, dtype=torch.float32, device=weight.device)

    # Launch enough programs to keep all SMs busy, each processing ~1K rows
    num_programs = min(N, 1024)  # RTX 5090 has 170 SMs; 1024 is more than enough
    grid = (num_programs,)

    embedding_kernel[grid](
        weight, idx, out,
        N, D,
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_D=D,                     # compile‑time constant, no mask needed
        num_warps=4,
        num_stages=2,
    )

    return out