#!/usr/bin/env python
"""Run one experiment arm.

Usage:
  python runner.py --config configs/arms/full.yaml --operators vector_add softmax
  python runner.py --config configs/arms/full.yaml --operators all
Overrides: --rounds N  --expand N  --total-candidates N  --model NAME  --gpus 0 1  --out DIR
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from harness.bandit import make_selector                      # noqa: E402
from harness.llm import LLMClient, set_scorer_system          # noqa: E402
from harness.memory import Journal, MetricsLog, StrategyMemory  # noqa: E402
from harness.prompts import SCORER_SYSTEM, system_prompt      # noqa: E402
from harness.search import OperatorSearch                     # noqa: E402
from harness.verifier import GPUPool                          # noqa: E402
from operators.specs import OPERATORS                         # noqa: E402


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(arm_path: str | None) -> dict:
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    if arm_path:
        cfg = deep_merge(cfg, yaml.safe_load(Path(arm_path).read_text()) or {})
    return cfg


async def run_arm(cfg: dict, operators: list[str], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    arm = cfg["arm"]

    set_scorer_system(SCORER_SYSTEM)
    llm = LLMClient(model=cfg["llm"]["model"],
                    reasoning_effort=cfg["llm"].get("reasoning_effort", "high"),
                    max_concurrent=cfg["llm"].get("max_concurrent", 10),
                    max_output_tokens=cfg["llm"].get("max_output_tokens", 24576),
                    temperature=cfg["llm"].get("temperature", 1.0))
    hw = cfg["hardware"]
    meta = await llm.setup(system_prompt(hw["description"]))

    conf_mode = cfg["confidence"].get("mode", "auto")
    if conf_mode == "auto":
        conf_mode = "logprobs" if meta.get("logprobs") else "verbalized"
    meta["confidence_mode_resolved"] = conf_mode

    kb = None
    kb_cfg = cfg.get("kb", {})
    if kb_cfg.get("use", False):
        from harness.kb import KB
        kb = KB.load(ROOT / kb_cfg.get("dir", "kb"))
        meta["kb_entries"] = len(kb)

    (out_dir / "run_meta.json").write_text(json.dumps(
        {"arm": arm, "operators": operators, "started": datetime.now().isoformat(),
         "llm": meta}, indent=2))
    print(f"[{arm}] llm capabilities: {meta}")

    pool = GPUPool(cfg["gpus"], timeout_s=cfg["budget"]["candidate_timeout_s"],
                   peak_gbps=hw["peak_gbps"], seed=cfg["seeds"]["input_seed"],
                   candidates_dir=out_dir / "candidates",
                   launch_stagger_s=cfg.get("launch_stagger_s", 0.75))
    journal = Journal(out_dir / "journal.jsonl")
    metrics = MetricsLog(out_dir / "metrics.csv")
    selector = make_selector(cfg["strategy"]["selector"], cfg["seeds"]["input_seed"])
    strat_mem = StrategyMemory({n: s.category for n, s in OPERATORS.items()})

    if kb is not None:
        print(f"[{arm}] kb loaded: {len(kb)} entries")

    t0 = time.time()
    searches = [OperatorSearch(OPERATORS[name], cfg, llm, pool, journal,
                               metrics, selector, strat_mem, arm, kb=kb)
                for name in operators]
    await asyncio.gather(*[s.run() for s in searches])

    (out_dir / "bandit_state.json").write_text(
        json.dumps(selector.snapshot(), indent=2))
    await llm.close()

    best = {}
    for s in searches:
        c = s._correct_nodes()
        best[s.spec.name] = max((n.result.speedup for n in c), default=0.0)
    summary = {"arm": arm, "wall_minutes": (time.time() - t0) / 60,
               "best_speedup": best,
               "total_candidates": sum(s.eval_counter for s in searches),
               "total_tokens": sum(s.cum_tokens for s in searches),
               "total_gpu_seconds": sum(s.cum_gpu for s in searches)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{arm}] done: {json.dumps(summary, indent=2)}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="arm yaml overlaying default.yaml")
    ap.add_argument("--operators", nargs="+", default=["all"])
    ap.add_argument("--rounds", type=int)
    ap.add_argument("--expand", type=int)
    ap.add_argument("--total-candidates", type=int)
    ap.add_argument("--model", help="override llm.model, e.g. deepseek-v4-pro")
    ap.add_argument("--gpus", nargs="+",
                    help="GPU ids, space or comma separated: --gpus 3 4 5 / --gpus 3,4,5")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.gpus:
        args.gpus = [int(g) for tok in args.gpus for g in tok.split(",") if g]

    cfg = load_config(args.config)
    if args.rounds:
        cfg["search"]["rounds"] = args.rounds
    if args.expand:
        cfg["search"]["expand_strategies"] = args.expand
    if args.total_candidates:
        cfg["budget"]["total_candidates"] = args.total_candidates
    if args.model:
        cfg["llm"]["model"] = args.model
    if args.gpus:
        cfg["gpus"] = args.gpus

    # budget-matched sequential baseline: same number of verified candidates
    if cfg["search"]["mode"] == "sequential":
        cfg["search"]["rounds"] = cfg["budget"]["total_candidates"]

    ops = list(OPERATORS) if args.operators == ["all"] else args.operators
    for o in ops:
        if o not in OPERATORS:
            sys.exit(f"unknown operator: {o} (known: {', '.join(OPERATORS)})")

    ts = datetime.now().strftime("%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else ROOT / "runs" / f"{cfg['arm']}_{ts}"
    asyncio.run(run_arm(cfg, ops, out_dir))


if __name__ == "__main__":
    main()
