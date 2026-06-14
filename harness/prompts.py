"""Prompt builders. The system prompt is stable (cacheable); per-candidate
context goes in the user turn."""
from __future__ import annotations

from harness.strategies import library_text, BY_ID
from operators.specs import OperatorSpec, ref_source


def system_prompt(hw_desc: str) -> str:
    return f"""You are an expert GPU kernel engineer writing Triton kernels.

TARGET HARDWARE: {hw_desc}

TASK: given a PyTorch reference operator, produce a complete Python module that
implements it with a Triton kernel, optimized for the target hardware.

HARD CONTRACT (violations are auto-rejected):
1. The module MUST define `def triton_run(*inputs) -> torch.Tensor` with the
   exact signature documented per operator. It allocates the output, launches
   the kernel(s), and returns the output tensor.
2. The core computation MUST be done in Triton kernels you write. Calling the
   equivalent PyTorch op (e.g. torch.softmax for the softmax task) is cheating
   and is rejected by a static checker. Using torch only for output allocation
   (torch.empty/empty_like) and trivial glue is fine.
3. No benchmarking, printing, or timing code in the module. The harness times
   `triton_run` externally with triton.testing.do_bench.
4. The module must be self-contained: only `import torch`, `import triton`,
   `import triton.language as tl`, `math`, and
   `from triton.language.extra import libdevice`. Top-level code runs at
   import time — keep it to definitions only.
5. Numerical tolerance per operator is given in the task. Stay within it.

OUTPUT FORMAT (exactly):
- One fenced python code block containing the full module.
- After the code block, a single line:
  CONF: {{"confidence": <0-100 integer>, "notes": "<one sentence>"}}
  where confidence is your honest estimate that the module compiles AND passes
  the correctness check on the first try.

{library_text()}

GUIDANCE:
- You will be assigned ONE strategy per attempt. Apply that strategy as the
  primary change; keep other aspects close to the parent version unless the
  strategy requires restructuring.
- Prefer correctness over aggressiveness: a fast-but-wrong kernel scores zero.
- Triton API level: Triton 3.x (tl.load/tl.store with masks, tl.dot,
  tl.constexpr, num_warps/num_stages launch kwargs, tl.exp2, tl.rsqrt, etc.).
- Common API pitfalls (Triton 3.6): there is NO tl.tanh and NO tl.math.tanh.
  For tanh use `from triton.language.extra import libdevice` then
  `libdevice.tanh(x)`, or the identity tanh(x) = 2*tl.sigmoid(2*x) - 1.
  erf exists as tl.math.erf (and libdevice.erf). tl.exp / tl.log / tl.sqrt /
  tl.rsqrt / tl.exp2 / tl.log2 / tl.sigmoid exist directly under tl.
  Allowed imports also include `from triton.language.extra import libdevice`.
"""


def _result_block(node) -> str:
    r = node.result
    if r is None:
        return "no result"
    if r.correct:
        return (f"correct=True  speedup={r.speedup:.3f}x  "
                f"triton={r.triton_ms:.4f}ms vs pytorch={r.pytorch_ms:.4f}ms  "
                f"achieved_bw={r.achieved_gbps:.0f}GB/s ({r.bw_utilization*100:.0f}% of peak)")
    return (f"correct=False  error_type={r.error_type}\n"
            f"error (tail of traceback — the last line is the actual error):\n"
            f"{r.error_msg[-1200:]}")


def generation_prompt(spec: OperatorSpec, strategy_id: str, parent,
                      branch_history: list[tuple[str, str]], round_idx: int,
                      kb_examples: list | None = None) -> str:
    """branch_history: [(strategy_id, outcome_summary), ...] along this branch.
    kb_examples: [(code, meta), ...] successful kernels from the knowledge base."""
    parts = [
        f"OPERATOR: {spec.name}  (category: {spec.category})",
        f"SIGNATURE: {spec.signature_doc}",
        f"TOLERANCE: rtol={spec.rtol}, atol={spec.atol}",
        "PYTORCH REFERENCE:",
        "```python",
        ref_source(spec).strip(),
        "```",
    ]
    if strategy_id in BY_ID:
        s = BY_ID[strategy_id]
        parts += [f"ASSIGNED STRATEGY: {s.id} — {s.snippet}"]
    elif strategy_id == "seed":
        parts += ["ASSIGNED STRATEGY: seed — write your best initial Triton "
                  "implementation. Choose sensible launch parameters; favor a "
                  "design with obvious follow-up tuning knobs."]
    else:
        parts += ["ASSIGNED STRATEGY: free refinement — analyze the feedback "
                  "below and make the single most promising improvement."]

    if parent is not None:
        parts += [
            f"\nPARENT VERSION (round {parent.round}, strategy={parent.strategy}):",
            "```python", parent.code.strip(), "```",
            "PARENT MEASUREMENT: " + _result_block(parent),
        ]
        if parent.bottleneck:
            parts += [f"PARENT ASSESSMENT: bottleneck={parent.bottleneck}, "
                      f"estimated headroom={parent.headroom_pct:.0f}%"]
    if branch_history:
        hist = "; ".join(f"{sid}→{outcome}" for sid, outcome in branch_history[-8:])
        parts += [f"\nALREADY TRIED ON THIS BRANCH: {hist}",
                  "Avoid repeating a failed direction unless you fix its root cause."]

    for code, meta in (kb_examples or []):
        same = "this operator" if meta["operator"] == spec.name \
            else f"a same-category operator ({meta['operator']})"
        parts += [
            f"\nREFERENCE KERNEL — a previously successful kernel for {same}, "
            f"measured {meta['speedup']:.2f}x at "
            f"{meta['bw_utilization']*100:.0f}% peak bandwidth. Reuse the "
            "working idioms (API usage, masking, launch setup); do not copy "
            "blindly if shapes/semantics differ:",
            "```python", code.strip(), "```",
        ]

    parts += [f"\nThis is round {round_idx}. Produce the module now, following "
              "the OUTPUT FORMAT exactly."]
    return "\n".join(parts)


SCORER_SYSTEM = """You are a GPU performance analyst. Given a Triton kernel,
its measurements, the history of attempts, and prior experience, estimate how
much speedup headroom remains and which strategies to try next. Be calibrated:
kernels near the memory-bandwidth roofline have ~0 headroom regardless of code
style.

Respond with a single JSON object only (no markdown), with exactly these keys:
{
  "headroom_pct": <number, estimated remaining speedup headroom in percent;
                   0 = already optimal, 50 = ~1.5x more plausible>,
  "bottleneck": <string, one of: memory_bandwidth | compute | launch_overhead
                 | occupancy | algorithm | unknown>,
  "suggested_strategies": <array of strategy id strings, best first>,
  "confidence": <integer 0-100>,
  "reasoning": <string, one short paragraph>
}"""


def scorer_prompt(spec: OperatorSpec, node, branch_history, memory_lines,
                  use_memory: bool, peak_gbps: float) -> str:
    r = node.result
    parts = [
        f"OPERATOR: {spec.name} (category: {spec.category}, "
        f"{'compute-bound' if spec.compute_bound else 'memory-bound'})",
        f"HARDWARE PEAK BANDWIDTH: {peak_gbps:.0f} GB/s",
        "KERNEL:",
        "```python", node.code.strip(), "```",
        "MEASUREMENT: " + _result_block(node),
    ]
    if branch_history:
        parts += ["BRANCH HISTORY (strategy → outcome): " +
                  "; ".join(f"{s}→{o}" for s, o in branch_history[-10:])]
    if use_memory and memory_lines:
        parts += ["PRIOR EXPERIENCE (this run, same category):"] + memory_lines
    parts += ["Estimate headroom_pct, the bottleneck, and the most promising "
              "strategy ids from the library: tune_block_size, tune_warps_stages, "
              "grid_loop_remap, constexpr_specialize, vectorized_access, "
              "multirow_per_program, cache_eviction_hints, mask_elimination, "
              "one_pass_welford, online_softmax, persistent_kernel, "
              "two_stage_reduction, kernel_fusion, tiling_shapes, group_swizzle, "
              "split_k, fast_math, precision_strategy, recompute_vs_store, "
              "atomic_free_combine."]
    return "\n".join(parts)
