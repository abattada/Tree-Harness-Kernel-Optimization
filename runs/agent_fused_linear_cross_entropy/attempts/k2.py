"""Fused linear cross-entropy (forward loss only) -- FP16 tensor-core variant.

Same split-LSE fusion as k0, but the matmul runs on FP16 tensor cores.
Rationale: TF32 and FP16 both carry a 10-bit mantissa, and all values here are
small and well inside FP16 range (x~N(0,1), w~0.02*N(0,1), logits~N(0,0.9)), so
FP16 gives the *same* accuracy as TF32 while delivering ~2x tensor throughput and
halving the w/x DRAM traffic. Accumulation stays FP32.
"""
import torch
import triton
import triton.language as tl

_MIN_BLOCK_N = 128


def _stage1_configs():
    cfgs = []
    # (bm, bn, bk, group_m, num_warps, num_stages)
    for bm, bn, bk, g, w, s in [
        (128, 128, 64, 8, 8, 4),
        (128, 128, 32, 8, 8, 4),
        (128, 256, 64, 8, 8, 3),
        (128, 256, 32, 8, 8, 4),
        (64, 256, 64, 8, 8, 3),
        (128, 128, 64, 8, 4, 4),
        (256, 128, 64, 8, 8, 3),
        (64, 128, 64, 4, 4, 4),
    ]:
        if bn < _MIN_BLOCK_N:
            continue
        cfgs.append(triton.Config(
            {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": g},
            num_warps=w, num_stages=s))
    return cfgs


@triton.autotune(configs=_stage1_configs(), key=["M", "N", "K"])
@triton.jit
def _flce_stage1(
    x_ptr, w_ptr, lse_part_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = w_ptr + offs_n[None, :] * K + offs_k[:, None]

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)          # fp16
        b = tl.load(b_ptrs)          # fp16
        acc += tl.dot(a, b, out_dtype=tl.float32)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K

    m_tile = tl.max(acc, axis=1)
    s_tile = tl.sum(tl.exp(acc - m_tile[:, None]), axis=1)
    block_lse = m_tile + tl.log(s_tile)
    tl.store(lse_part_ptr + pid_n * M + offs_m, block_lse)


@triton.jit
def _flce_rowloss(
    x_ptr, w_ptr, tgt_ptr, lse_part_ptr, loss_ptr,
    M, K,
    NPN: tl.constexpr, BK: tl.constexpr,
):
    m = tl.program_id(0)
    offs = tl.arange(0, NPN)
    lse_vals = tl.load(lse_part_ptr + offs * M + m)
    gmax = tl.max(lse_vals, axis=0)
    row_lse = gmax + tl.log(tl.sum(tl.exp(lse_vals - gmax), axis=0))

    t = tl.load(tgt_ptr + m)
    acc = tl.zeros([], dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        xv = tl.load(x_ptr + m * K + offs_k).to(tl.float32)
        wv = tl.load(w_ptr + t * K + offs_k).to(tl.float32)
        acc += tl.sum(xv * wv, axis=0)

    tl.store(loss_ptr + m, row_lse - acc)


@triton.jit
def _flce_meansum(loss_ptr, out_ptr, M, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < M
    v = tl.load(loss_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr, tl.sum(v, axis=0) / M)


def triton_run(x: torch.Tensor, w: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    M, K = x.shape
    V, Kw = w.shape
    N = V
    NPN = N // _MIN_BLOCK_N

    xh = x.to(torch.float16)
    wh = w.to(torch.float16)

    lse_part = torch.full((NPN, M), float("-inf"), device=x.device, dtype=torch.float32)
    loss = torch.empty(M, device=x.device, dtype=torch.float32)
    out = torch.empty((), device=x.device, dtype=torch.float32)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    _flce_stage1[grid](xh, wh, lse_part, M, N, K)

    _flce_rowloss[(M,)](xh, wh, targets, lse_part, loss, M, K, NPN=NPN, BK=512)

    block = triton.next_power_of_2(M)
    _flce_meansum[(1,)](loss, out, M, BLOCK=block)
    return out
