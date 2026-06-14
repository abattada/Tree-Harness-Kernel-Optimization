"""Strategy library: named optimization directions the LLM is asked to apply.

Each strategy has an id, the op categories it applies to, and a prompt snippet.
The full library text is embedded in the (cached) system prompt; per-candidate
prompts reference one assigned strategy.
"""
from __future__ import annotations

from dataclasses import dataclass

ALL = {"elementwise", "reduction", "normalization", "loss", "matmul", "misc"}


@dataclass(frozen=True)
class Strategy:
    id: str
    kind: str                # launch | memory | structure | numeric
    applies: frozenset
    snippet: str


def _s(id, kind, applies, snippet):
    return Strategy(id, kind, frozenset(applies), snippet)


STRATEGIES: list[Strategy] = [
    _s("tune_block_size", "launch", ALL,
       "Sweep BLOCK sizes (and 2D tile shapes). Pick the value that balances "
       "occupancy vs. register pressure; powers of two from 128 to 8192 for 1D."),
    _s("tune_warps_stages", "launch", ALL,
       "Tune num_warps (1..16) and num_stages (2..6) for the chosen block size. "
       "Small rows want fewer warps; pipelined loads want more stages."),
    _s("grid_loop_remap", "launch", {"elementwise", "reduction", "misc"},
       "Launch fewer programs, each looping over multiple tiles "
       "(grid-stride loop) to amortize launch and scheduling overhead."),
    _s("constexpr_specialize", "launch", ALL,
       "Specialize shapes as tl.constexpr and use tl.static_assert / "
       "tl.multiple_of / tl.max_contiguous hints so the compiler can "
       "vectorize and drop boundary masks."),

    _s("vectorized_access", "memory", ALL,
       "Make every load/store wide and coalesced: contiguous tl.arange ranges, "
       "process several elements per thread, align block boundaries to 128B."),
    _s("multirow_per_program", "memory", {"normalization", "reduction", "loss"},
       "Process multiple rows per program (2D block) so short rows stop "
       "underutilizing the SM; tune ROWS_PER_PROG."),
    _s("cache_eviction_hints", "memory", ALL,
       "Use cache modifiers/eviction policies (e.g. eviction_policy='evict_first' "
       "for streaming inputs, 'evict_last' for reused tiles)."),
    _s("mask_elimination", "memory", {"elementwise", "normalization", "misc"},
       "When sizes divide the block evenly, drop masks entirely (or split the "
       "grid into a full-tile fast path + boundary path)."),

    _s("one_pass_welford", "structure", {"reduction", "normalization"},
       "Replace two-pass mean/variance with a single-pass Welford or "
       "sum/sumsq computation to halve memory traffic."),
    _s("online_softmax", "structure", {"normalization", "loss"},
       "Use the online (single-pass) softmax/logsumexp recurrence to fuse the "
       "max and sum passes and reduce reads of the row."),
    _s("persistent_kernel", "structure", {"reduction", "matmul", "loss"},
       "Use a persistent-kernel formulation: #programs = #SMs, each program "
       "loops over work items; combine partial results with one final pass "
       "(avoid atomics where possible)."),
    _s("two_stage_reduction", "structure", {"reduction", "loss"},
       "Tune the two-stage reduction split: partial-sum count and per-stage "
       "block sizes. Stage 2 must also be a Triton kernel — calling torch "
       "reduction APIs (.sum()/torch.sum) for the final combine is rejected."),
    _s("kernel_fusion", "structure", {"loss", "normalization", "misc", "elementwise"},
       "Fuse adjacent computations into one kernel so intermediates stay in "
       "registers (e.g. matmul tile + logsumexp + NLL for fused CE)."),
    _s("tiling_shapes", "structure", {"matmul"},
       "Tune BLOCK_M/BLOCK_N/BLOCK_K for tensor-core shapes (multiples of 16); "
       "try wide-N vs square tiles; ensure K-loop is pipelined."),
    _s("group_swizzle", "structure", {"matmul"},
       "Reorder program ids in grouped/swizzled order (GROUP_M) to improve L2 "
       "reuse across neighboring tiles."),
    _s("split_k", "structure", {"matmul"},
       "Use split-K (parallelize the K dimension across programs, then reduce) "
       "when M*N is small relative to K or occupancy is low."),

    _s("fast_math", "numeric", {"elementwise", "normalization", "loss"},
       "Use faster math: exp2/log2 with rescaling instead of exp/log, rsqrt "
       "instead of 1/sqrt, fdiv fast paths — keep within tolerance."),
    _s("precision_strategy", "numeric", {"matmul"},
       "Tune tl.dot input_precision / accumulator dtype (fp16 in, fp32 acc) "
       "and output converts; ensure tensor cores are engaged."),
    _s("recompute_vs_store", "numeric", {"normalization", "loss", "reduction"},
       "Trade recomputation for memory traffic: keep row statistics in "
       "registers and recompute cheap values instead of re-reading DRAM."),
    _s("atomic_free_combine", "structure", {"reduction", "loss"},
       "Avoid global atomics: write per-program partials and combine in a "
       "second tiny kernel or with a deterministic tree reduction."),
]

BY_ID = {s.id: s for s in STRATEGIES}
SEED_STRATEGY = "seed"          # round-0 pseudo-strategy (not in the library)
SEQ_STRATEGY = "sequential_refine"


def applicable(op_category: str) -> list[Strategy]:
    return [s for s in STRATEGIES if op_category in s.applies]


def library_text() -> str:
    lines = ["STRATEGY LIBRARY (id | kind | applies | description):"]
    for s in STRATEGIES:
        lines.append(f"- {s.id} | {s.kind} | {','.join(sorted(s.applies))} | {s.snippet}")
    return "\n".join(lines)
