"""Tree / sequential search loops (the closed loop of the harness)."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from harness.bandit import BaseSelector
from harness.memory import Journal, MetricsLog, StrategyMemory
from harness.models import EvalResult, Node
from harness.prompts import generation_prompt, scorer_prompt
from harness.scoring import apply_score
from harness.strategies import BY_ID, SEED_STRATEGY, SEQ_STRATEGY, applicable
from harness.verifier import GPUPool
from operators.specs import OperatorSpec


SEED_HINTS = [
    "favor a straightforward design with clean tuning knobs",
    "favor wide vectorized memory access",
    "favor a design that minimizes the number of passes over the data",
    "favor higher occupancy (smaller blocks, more programs)",
    "favor fewer, larger programs that loop over tiles",
    "favor a layout that keeps reused values in registers",
    "favor aggressive constexpr specialization to the exact shapes",
    "favor the simplest correct kernel imaginable",
]


class OperatorSearch:
    def __init__(self, spec: OperatorSpec, cfg: dict, llm, pool: GPUPool,
                 journal: Journal, metrics: MetricsLog, selector: BaseSelector,
                 strat_mem: StrategyMemory, arm: str, kb=None):
        self.spec = spec
        self.cfg = cfg
        self.llm = llm
        self.pool = pool
        self.journal = journal
        self.metrics = metrics
        self.selector = selector
        self.mem = strat_mem
        self.arm = arm
        self.kb_examples = (kb.exemplars_for(
            spec.name, spec.category,
            k=cfg.get("kb", {}).get("max_exemplars", 2)) if kb else [])

        tcfg = cfg.get("transcripts", {})
        self.transcripts_on = tcfg.get("enabled", True)
        self.transcripts_reasoning = tcfg.get("include_reasoning", True)
        self.transcripts_dir = (Path(journal.path).parent / "transcripts"
                                if self.transcripts_on else None)

        self.nodes: dict[str, Node] = {}
        self.eval_counter = 0
        self.budget = int(cfg["budget"]["total_candidates"])
        self.deadline = time.time() + cfg["budget"]["minutes_per_operator"] * 60
        self.cum_gpu = 0.0
        self.cum_tokens = 0

    # ------------------------------------------------------------- helpers

    def _remaining(self) -> int:
        return self.budget - self.eval_counter

    def _out_of_time(self) -> bool:
        return time.time() > self.deadline

    def _correct_nodes(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.result and n.result.correct]

    def _best_speedup(self) -> float:
        c = self._correct_nodes()
        return max((n.result.speedup for n in c), default=0.0)

    def _branch_history(self, node: Node | None) -> list[tuple[str, str]]:
        hist = []
        cur = node
        while cur is not None:
            r = cur.result
            outcome = (f"{r.speedup:.2f}x" if r and r.correct
                       else (r.error_type if r else "pending"))
            hist.append((cur.strategy, outcome))
            cur = self.nodes.get(cur.parent_id) if cur.parent_id else None
        return list(reversed(hist))

    def _tried_on_branch(self, node: Node | None) -> set[str]:
        return {sid for sid, _ in self._branch_history(node)}

    def _write_transcript(self, node: Node, kind: str, payload: dict):
        """Persist one LLM exchange (prompt + raw response [+ reasoning]) for the
        report. One JSON per call; no journal-schema change. kind: generation |
        assessment."""
        if not self.transcripts_dir or not payload.get("prompt"):
            return
        rec = {
            "node_id": node.node_id, "arm": self.arm, "operator": self.spec.name,
            "round": node.round, "eval_index": node.eval_index,
            "strategy": node.strategy, "kind": kind,
            "error": payload.get("error", ""),
            "usage": payload.get("usage", {}),
            "system": payload.get("system", ""),
            "prompt": payload.get("prompt", ""),
            "response": payload.get("raw_response", ""),
        }
        if self.transcripts_reasoning:
            rec["reasoning"] = payload.get("reasoning", "")
        try:
            self.transcripts_dir.mkdir(parents=True, exist_ok=True)
            path = self.transcripts_dir / f"{node.eval_index:04d}_{node.node_id}_{kind}.json"
            path.write_text(json.dumps(rec, ensure_ascii=False, indent=2,
                                       default=str))
        except OSError:
            pass

    # --------------------------------------------------------- gen + eval

    async def _make_node(self, strategy: str, parent: Node | None,
                         round_idx: int, seed_hint: str = "") -> Node:
        t0 = time.time()
        prompt = generation_prompt(self.spec, strategy, parent,
                                   self._branch_history(parent), round_idx,
                                   kb_examples=self.kb_examples)
        if seed_hint:
            prompt += f"\nDESIGN HINT for this variant: {seed_hint}"

        node = Node(operator=self.spec.name, round=round_idx, strategy=strategy,
                    code="", parent_id=parent.node_id if parent else None)
        gen = await self.llm.generate_candidate(prompt)
        u = gen.get("usage", {}) or {}
        node.input_tokens = u.get("input_tokens", 0)
        node.output_tokens = u.get("output_tokens", 0)
        node.cached_tokens = u.get("cached_tokens", 0)
        node.thought_tokens = u.get("thought_tokens", 0)
        self.cum_tokens += node.input_tokens + node.output_tokens

        node.eval_index = self.eval_counter
        self.eval_counter += 1
        self._write_transcript(node, "generation", gen)

        if "code" not in gen:
            node.result = EvalResult(ok=False, error_type="genfail",
                                     error_msg=gen.get("error", "")[:500])
        else:
            node.code = gen["code"]
            node.llm_confidence = gen.get("llm_confidence", -1.0)
            node.logprob_conf = gen.get("logprob_conf")
            node.gen_notes = gen.get("notes", "")
            node.result = await self.pool.evaluate(
                self.spec.name, node.code,
                tag=f"{self.arm}_{self.spec.name}_{node.node_id}")
            self.cum_gpu += node.result.gpu_seconds

        node.wall_time = time.time() - t0
        self.nodes[node.node_id] = node
        return node

    async def _assess(self, node: Node):
        scfg = self.cfg["scorer"]
        if not scfg.get("enabled", True):
            return
        if not (node.result and node.result.correct):
            return
        mem_lines = (self.mem.lines_for(self.spec.category)
                     if scfg.get("use_memory", True) else [])
        res = await self.llm.assess_node(scorer_prompt(
            self.spec, node, self._branch_history(node), mem_lines,
            scfg.get("use_memory", True), self.cfg["hardware"]["peak_gbps"]))
        self._write_transcript(node, "assessment", res)
        if "assessment" in res:
            a = res["assessment"]
            node.headroom_pct = max(0.0, float(a.headroom_pct))
            node.bottleneck = a.bottleneck
            node.suggested_strategies = [s for s in a.suggested_strategies
                                         if s in BY_ID][:6]
            node.assess_confidence = a.confidence
        u = res.get("usage", {}) or {}
        node.input_tokens += u.get("input_tokens", 0)
        node.output_tokens += u.get("output_tokens", 0)
        self.cum_tokens += u.get("input_tokens", 0) + u.get("output_tokens", 0)

    def _finalize_round(self, new_nodes: list[Node], round_idx: int,
                        beam: list[Node]):
        score_cfg = dict(self.cfg["scorer"])
        score_cfg["conf_in_score"] = self.cfg["confidence"].get("use_in_score", False)
        beam_ids = {n.node_id for n in beam}
        for n in new_nodes:
            apply_score(n, score_cfg, self.spec.compute_bound)
            n.selected_into_beam = n.node_id in beam_ids
            row = n.journal_row(self.arm)
            self.mem.observe(row)
            if n.strategy in BY_ID:
                self.selector.update(self.spec.category, n.strategy,
                                     self.mem.relative_reward(row))
            self.journal.append(row)
        self.metrics.append(self.arm, self.spec.name, round_idx,
                            self.eval_counter, self._best_speedup(),
                            len(self._correct_nodes()), self.cum_gpu,
                            self.cum_tokens)

    def _select_beam(self, k: int) -> list[Node]:
        score_cfg = dict(self.cfg["scorer"])
        score_cfg["conf_in_score"] = self.cfg["confidence"].get("use_in_score", False)
        cands = self._correct_nodes()
        for n in cands:
            apply_score(n, score_cfg, self.spec.compute_bound)
        cands.sort(key=lambda n: n.score_final, reverse=True)
        return cands[:k]

    # -------------------------------------------------------------- modes

    async def run(self):
        mode = self.cfg["search"]["mode"]
        if mode == "tree":
            await self._run_tree()
        elif mode == "sequential":
            await self._run_sequential()
        else:
            raise ValueError(f"unknown search.mode: {mode}")

    async def _run_tree(self):
        s = self.cfg["search"]
        beam_k = int(s["beam_k"])
        n_strats = int(s["expand_strategies"])
        n_impls = int(s["impls_per_strategy"])
        rounds = int(s["rounds"])

        # round 0: seeds
        n_seed = min(beam_k * n_impls, self._remaining())
        seeds = await asyncio.gather(*[
            self._make_node(SEED_STRATEGY, None, 0,
                            seed_hint=SEED_HINTS[i % len(SEED_HINTS)])
            for i in range(n_seed)])
        await asyncio.gather(*[self._assess(n) for n in seeds])
        beam = self._select_beam(beam_k)
        self._finalize_round(list(seeds), 0, beam)

        for rnd in range(1, rounds + 1):
            if self._remaining() <= 0 or self._out_of_time():
                break
            if not beam:
                # repair mode: nothing correct yet. Optimizing broken code is
                # useless — run free error-fix refinement on the latest
                # failures (same prompt as the sequential baseline) until at
                # least one candidate passes, then resume strategy expansion.
                failed = sorted(self.nodes.values(),
                                key=lambda n: n.eval_index)[-beam_k:]
                jobs = [(SEQ_STRATEGY, p) for p in failed
                        for _ in range(n_impls)]
                jobs = jobs[:self._remaining()]
                if not jobs:
                    break
                new_nodes = await asyncio.gather(*[
                    self._make_node(sid, parent, rnd) for sid, parent in jobs])
                await asyncio.gather(*[self._assess(n) for n in new_nodes])
                beam = self._select_beam(beam_k)
                self._finalize_round(list(new_nodes), rnd, beam)
                continue
            jobs = []
            for bnode in beam:
                tried = self._tried_on_branch(bnode)
                picks = []
                if self.cfg["scorer"].get("enabled", True) and \
                        self.cfg["scorer"].get("use_suggestions", True):
                    picks = [sid for sid in bnode.suggested_strategies
                             if sid not in tried][:n_strats]
                pool_ids = {p for p in picks}
                extra = self.selector.select(self.spec.category,
                                             n_strats - len(picks),
                                             exclude=tried | pool_ids)
                picks += extra
                if not picks:
                    # no strategy assignment (selector=none, scorer off):
                    # tree_free — branch with free refinement at full width
                    picks = [SEQ_STRATEGY] * n_strats
                for sid in picks[:n_strats]:
                    for _ in range(n_impls):
                        jobs.append((sid, bnode))
            jobs = jobs[:self._remaining()]
            if not jobs:
                break
            new_nodes = await asyncio.gather(*[
                self._make_node(sid, parent, rnd) for sid, parent in jobs])
            await asyncio.gather(*[self._assess(n) for n in new_nodes])
            beam = self._select_beam(beam_k)
            self._finalize_round(list(new_nodes), rnd, beam)

    async def _run_sequential(self):
        """Single-chain refinement baseline; budget-matched to the tree arms
        (one candidate per round, rounds = total_candidates)."""
        parent: Node | None = None
        best: Node | None = None
        rnd = 0
        while self._remaining() > 0 and not self._out_of_time():
            rnd += 1
            strategy = SEED_STRATEGY if parent is None else SEQ_STRATEGY
            node = await self._make_node(strategy, parent, rnd)
            await self._assess(node)
            if node.result and node.result.correct:
                if best is None or node.result.speedup >= best.result.speedup:
                    best = node
            # refine from the best correct node so far; before any success,
            # iterate on the latest failed attempt (error-fix loop)
            parent = best if best is not None else node
            self._finalize_round([node], rnd, [best] if best else [])
