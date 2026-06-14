# STATE — fused_linear_cross_entropy
status: saturated      # triton_ms ~1.84ms is at the fp16 GEMM hardware floor + unavoidable cast
# best / budget 不在這裡手寫:看 `python -m harness.task fused_linear_cross_entropy` 任務卡。

## tried(每行一筆,新的加在最上面;格式: k<N> | <strategy> | <speedup 或 error 根因>)
- k6 | fp16acc_tight_confirm | 5.11x (triton 1.84ms, win BM256/BN256/BK32/GROUP16/s3) — re-confirm, at floor
- k5 | fp16acc_tight_grid | 4.98x (triton 1.83ms = best kernel; ratio dipped only b/c pytorch sample was 9.1ms)
- k4 | fp16acc_bn256_fixed | 5.15x (triton 2.04ms) ← BEST RECORDED. fix BLOCK_N=256 → kills k3 autotune bug
- k3 | fp16_acc_bigtile | WRONG (err 0.405): autotune shares one lse_part across trial configs; mixed BLOCK_N → stale slots double-counted
- k2 | fp16_tensorcore | 3.44x (triton 2.70ms; fp16 inputs but fp32 accumulate)
- k1 | tile_sweep | 1.88x — autotune mis-picked BM256 (noise), slower than k0
- k0 | fused_tf32_splitlse | 1.94x (triton 4.84ms; TF32 tensor cores, fp32 acc, split-LSE fusion baseline)

## pitfalls(本 op 踩過的環境/API 坑)
- triton.autotune reuses the candidate's allocated buffers across ALL trial configs. If a partial
  buffer's written-slot set depends on a tuned param (here BLOCK_N → num_pid_n), a wide-tile trial
  leaves STALE values where a narrow trial wrote, corrupting a later reduction (k3: err 0.405 even
  though every config is correct in isolation). FIX: fix BLOCK_N so all configs write the identical
  slot set. Verify correctness via eval, not just local single-config runs.
- Autotune self-pick is noisy (~10%); k1 picked a slower config than k0 despite k0's config being in
  the grid. Find the true-best config with a LOCAL do_bench sweep (free, no budget) and tighten the
  autotune grid around it.
- In-kernel fp32->fp16 cast is SLOWER than a separate torch .half() pass: fp32 w is re-read ~16x
  (2x DRAM traffic) and 256x256 tiles OOM shared memory. Pre-cast to fp16 buffers in triton_run.
- tl.tanh / tl.math.tanh absent (pre-existing) — n/a here.

## key facts (machine can't recompute these — read before iterating)
- Reference = F.cross_entropy(x @ w.t()) in FULL fp32 (allow_tf32 default False); ~9.1-10.5ms, NOISY.
  speedup ratio swings 4.98-5.15x almost entirely from this reference variance.
- tol=1e-2 is loose. FP16 and TF32 share a 10-bit mantissa; values are tiny (w*0.02 → logits~N(0,0.9))
  so FP16 == TF32 accuracy. Pure FP16 ACCUMULATE of the K=2048 contraction passes with err ~9e-6.
- Consumer Blackwell (5090): fp16+fp32-acc ≈232 TFLOPS (== cuBLAS for this shape), fp16+fp16-acc
  runs full-rate ≈364 TFLOPS. cuBLAS fp16 itself = 2.37ms; our Triton GEMM (1.51ms) BEATS it.
- GEMM floor for this shape ≈1.51ms and PLATEAUS across every tile config tried (256x256/128x256/...)
  → hardware-limited, not a tuning problem.
- 1.84ms breakdown: torch fp16 cast 0.29 | stage1 GEMM 1.51 | rowloss 0.04 | meansum 0.005.
- Design: split-LSE fusion. stage1 = grouped fp16 GEMM whose epilogue reduces each logit tile to a
  per-tile log-sum-exp into a tiny [N/256, M] buffer (logits NEVER materialised). rowloss combines
  the tile-LSEs per row and subtracts the target logit. meansum = scalar mean.

## next(按優先序)
1. NOT RECOMMENDED to spend more budget on speedup. We are at the fp16 GEMM hardware floor; remaining
   spread is reference-timing noise. Evidence: cuBLAS fp16 = 2.37ms (we beat it at 1.51), GEMM time is
   flat across all tile shapes, in-kernel-cast tested slower, fp8 rejected (3-bit mantissa → logit err
   ~0.1 >> 1e-2).
2. Only structural lever left (~0.29ms = 16%): a streaming/persistent kernel that reads w (fp32) from
   DRAM exactly once with no re-reads, removing the separate fp16 cast pass. Hard (needs cross-block
   w reuse); per-tile in-kernel cast already proven slower. Upside if solved: ~1.55ms → ~6x. Low conf.
