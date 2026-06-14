# STATE — gemm
status: saturated      # at the cuBLAS f32-acc frontier; the only faster path (f16-acc) is precluded by tolerance
# best / budget_used 不在這裡手寫:看 `python -m harness.task gemm`(從 journal 自動算)。
# best.py 用 `python -m harness.task gemm --export-best` 還原(目前 = k10 的 tight grid,1.011x)。

# KEY RESULT: f32-accumulate is the only correct path, and it now slightly BEATS cuBLAS
# (~1.01x) using Blackwell thread-block clusters (num_ctas=2) + shallow pipeline (num_stages=3).
# Winning config (converged across k8/k9/k10/k11):
#   BLOCK_M=128 BLOCK_N=128 BLOCK_K=64 GROUP_M=8 num_warps=8 num_stages=3 num_ctas=2
#
# NOTE on numbering: attempt files are k0..k11, but the journal has 14 rows because
# k0 and k1 were each evaluated twice (first run cheat-failed on the regex below, then
# the corrected file was re-run). So journal eval_index != attempt-file index. The next
# agent should write attempts/k14.py (next eval_index) and keep using private journals
# per the parallel-run note in pitfalls.

## tried(每行一筆,新的加在最上面;格式: k<file> | <strategy> | <結果>)
- k11 | probe_ctas4_bigtile      | 1.006x — num_ctas=4 / 128x256 / 256x128 / BK=128 all LOSE to ctas2 128x128x64 s3
- k10 | tight_winner_refine      | 1.011x BEST — winner 128x128x64/GM8/w8/s3/ctas2 (stages2/4, GM4/16, BK32/128 all slower)
- k9  | cluster_groupm_stages    | 1.009x — same winner; GROUP_M 8 best (4/16 worse), stages 3 best
- k8  | cluster_ctas24_sweep     | 0.995x — num_ctas=2 + num_stages=3 is the combo; num_ctas=4 no better
- k7  | clusters_num_ctas        | 0.957x — num_ctas=2 BEATS num_ctas=1 fallback → clusters are THE lever
- k6  | persistent_f32acc        | 0.919x — persistent (grid=#SMs, tile-loop) is WORSE than the simple 1-tile-per-CTA grid here
- k5  | f32acc_groupm_bk128      | 0.942x — GROUP_M sweep {1,4,8,16} + BLOCK_K=128: no help (picks GM=1 by noise)
- k4  | f32acc_nomod_sweep       | 0.943x — dropping the %M/%N modulo + wide tile/stage sweep: no help vs k0
- k3  | f16mma_f32flush_bk64     | wrong_output: mean_abs_err 0.0167 > atol 0.02 (f16 product-rounding floor)
- k2  | f16mma_f32flush_bk32     | wrong_output: mean_abs_err 0.0130, max 0.25 — better but tail elems still > atol
- k1  | f16acc_swizzle (rerun)   | wrong_output 1.507x — full-K f16 accumulate, mean_abs_err 0.105 (fast but unusable)
- k0  | f32acc_swizzle (rerun)   | 0.947x — first correct baseline (winning 128x128x32/w4); == prior SOTA
- k1  | f16acc_swizzle (1st run) | cheat (false positive, see pitfalls)
- k0  | f32acc_swizzle (1st run) | cheat (false positive, see pitfalls)

## pitfalls(本 op 踩過的環境/API 坑,下一個 agent 不要再踩)
- CHEAT-CHECK FALSE POSITIVE: the matmul forbidden regex `[\w\)\]]\s*@\s*[\w\(]` spans
  newlines, so it flags ANY decorator preceded by code — e.g. `]` (end of a configs list)
  or `)` (a stacked `@triton.autotune(...)`) followed by `@triton.jit`. Fix: use a SINGLE
  `@triton.jit`, put a `}`-ending sentinel line (`_S = {}`) right before it, and apply
  autotune as a normal call AFTER the def: `k = triton.autotune(cfgs, key=...)(k)`.
  Also keep any literal `X] @ Y` out of code (docstrings ARE stripped, so they're fine).
- f16-ACCUMULATE IS PERMANENTLY OUT for this op. tl.dot(out_dtype=tl.float16) runs ~1.5x
  faster (2x tensor-core roofline) but the f16 product/accumulator rounding gives an abs
  noise floor of ~sqrt(K)*eps_f16 ≈ 0.03 on near-zero output elements (output ~ N(0,K),
  K=4096). atol=0.02 < that, and strict torch.allclose needs ALL 16M elements to pass.
  Flushing every BLOCK_K to f32 (k2/k3) only cuts the chain length; the per-product f16
  rounding still leaves ~5σ ≈ 0.06 tail error even at BLOCK_K=16. Don't re-try f16-acc.
- AUTOTUNE PICKS BY SPEED, NOT CORRECTNESS (HARNESS rule 4). Anything that changes numerics
  (accumulator dtype, flush frequency / BLOCK_K when flushing) must NOT vary inside one grid,
  or autotune may pick a fast-but-wrong config and the whole eval fails. Keep such variants
  in separate candidate files.
- PARALLEL EVALS: ran 2 candidates at once on different GPUs, each writing its OWN journal
  (/tmp/gemm_jrnl/kN.jsonl), then cat-merged into runs/agent_gemm/journal.jsonl in order.
  This avoids the line-count race / large-row interleaving in eval_one's append. GPUs 0-3
  were idle; GPU 4 was ~99% busy (would corrupt timing) — avoid busy GPUs for benchmarking.
- One full eval (16-config autotune + correctness + do_bench) is only ~6-14 GPU-seconds, well
  under the 120s timeout, so ~16-config grids are safe and you can run several in parallel.

## next(給下一個 agent 的建議,按優先序)
1. NOT RECOMMENDED to keep climbing on raw config tuning — SATURATED. Evidence: k8/k9/k10/k11
   (~40 configs spanning num_ctas{1,2,4}, tiles{64..256}, BLOCK_K{32,64,128}, GROUP_M{1,4,8,16},
   num_warps{4,8}, num_stages{2,3,4,5}) ALL converge on 128x128x64/GM8/w8/s3/ctas2 at
   1.0-1.01x. cuBLAS sits at the f32-acc roofline (~224 TFLOPS); we're ~1% past it and the
   remaining run-to-run variation (~1%) is measurement noise, not headroom.
2. If you must spend budget: the only unexplored ideas with any (small, uncertain) upside are
   TMA block-pointer loads (`tl.make_block_ptr` / device tensor descriptors) and a
   cluster-aware persistent kernel that combines k6+k7 — but k6 (persistent alone) was a
   regression, so expect <=1-2% and likely noise. Re-confirm best.py reproduces >1.0x first.
3. Do NOT re-attempt f16-accumulate in any form (see pitfalls) — it is tolerance-blocked.
