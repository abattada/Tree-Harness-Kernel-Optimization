import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice  # not used here, but imported for completeness


@triton.jit
def _welford_kernel(
    x_ptr,
    out_ptr,
    N_ROWS: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    row_start = row_id * N_COLS
    tid = tl.arange(0, BLOCK_SIZE)

    # per-thread Welford accumulators
    count = tl.full((BLOCK_SIZE,), 0, dtype=tl.int32)
    mean = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    M2 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # iterate over columns in blocks
    for block_start in range(0, N_COLS, BLOCK_SIZE):
        cols = block_start + tid
        mask = cols < N_COLS
        vals = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0)

        valid = mask.to(tl.int32)  # 1 for valid, 0 otherwise
        count_new = count + valid
        delta = vals - mean
        # update mean and M2 only where element is valid
        mean_new = tl.where(
            valid != 0,
            mean + delta / count_new.to(tl.float32),
            mean,
        )
        delta2 = vals - mean_new
        M2_new = tl.where(
            valid != 0,
            M2 + delta * delta2,
            M2,
        )
        count = count_new
        mean = mean_new
        M2 = M2_new

    # parallel reduction across threads in a block using shared memory
    s_cnt = tl.static_shared(shape=(BLOCK_SIZE,), dtype=tl.int32)
    s_mean = tl.static_shared(shape=(BLOCK_SIZE,), dtype=tl.float32)
    s_M2 = tl.static_shared(shape=(BLOCK_SIZE,), dtype=tl.float32)

    ptr_cnt = s_cnt.to(tl.pointer_type(tl.int32))
    ptr_mean = s_mean.to(tl.pointer_type(tl.float32))
    ptr_M2 = s_M2.to(tl.pointer_type(tl.float32))

    tl.store(ptr_cnt + tid, count)
    tl.store(ptr_mean + tid, mean)
    tl.store(ptr_M2 + tid, M2)
    tl.debug_barrier()

    stride = BLOCK_SIZE // 2
    while stride > 0:
        if tid < stride:
            neighbor = tid + stride
            n_cnt = tl.load(ptr_cnt + neighbor)
            n_mean = tl.load(ptr_mean + neighbor)
            n_M2 = tl.load(ptr_M2 + neighbor)

            c_cnt = tl.load(ptr_cnt + tid)
            c_mean = tl.load(ptr_mean + tid)
            c_M2 = tl.load(ptr_M2 + tid)

            total = c_cnt + n_cnt
            delta = n_mean - c_mean
            # avoid division by zero (safe because at least one row has N_COLS > 0)
            total_float = total.to(tl.float32)
            inv_total = tl.where(total > 0, 1.0 / total_float, 0.0)
            new_mean = c_mean + delta * (n_cnt.to(tl.float32)) * inv_total
            new_M2 = (
                c_M2
                + n_M2
                + delta
                * delta
                * (c_cnt.to(tl.float32))
                * (n_cnt.to(tl.float32))
                * inv_total
            )
            tl.store(ptr_cnt + tid, total)
            tl.store(ptr_mean + tid, new_mean)
            tl.store(ptr_M2 + tid, new_M2)

        stride //= 2
        tl.debug_barrier()

    # thread 0 writes the final per-row statistics
    if tid == 0:
        final_cnt = tl.load(ptr_cnt + 0)
        final_mean = tl.load(ptr_mean + 0)
        final_M2 = tl.load(ptr_M2 + 0)
        variance = tl.where(final_cnt > 0, final_M2 / final_cnt.to(tl.float32), 0.0)
        tl.store(out_ptr + row_id, final_mean)          # row 0: mean
        tl.store(out_ptr + N_ROWS + row_id, variance)   # row 1: variance


def triton_run(x: torch.Tensor) -> torch.Tensor:
    N_ROWS, N_COLS = x.shape
    assert N_ROWS == 8192 and N_COLS == 4096, "Expected shape (8192, 4096)"
    out = torch.empty(2, N_ROWS, device=x.device, dtype=x.dtype)
    BLOCK_SIZE = 1024
    grid = (N_ROWS,)
    _welford_kernel[grid](
        x,
        out,
        N_ROWS,
        N_COLS,
        BLOCK_SIZE,
        num_warps=32,
    )
    return out