#!/usr/bin/env python
"""Per-candidate token statistics + cost extrapolation from pilot journals.

  python analysis/token_report.py --runs runs/pilot_*/pilot_full runs/pilot_*/pilot_seq \
      --target-candidates 544 --ablation-candidates 672
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# deepseek-v4-flash pricing (USD per 1M tokens), 2026-06
PRICE_IN_MISS = 0.14
PRICE_IN_HIT = 0.0028
PRICE_OUT = 0.28


def load(run_dirs):
    rows = []
    for d in run_dirs:
        p = Path(d) / "journal.jsonl"
        if not p.exists():
            print(f"[warn] missing {p}")
            continue
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                r.pop("code", None)
                rows.append(r)
    return pd.DataFrame(rows)


def cost_of(n_cand, per):
    miss = per["input_miss"] * n_cand / 1e6 * PRICE_IN_MISS
    hit = per["cached"] * n_cand / 1e6 * PRICE_IN_HIT
    out = per["output"] * n_cand / 1e6 * PRICE_OUT
    return miss + hit + out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--target-candidates", type=int, default=544,
                    help="main experiment size (17 ops x 32)")
    ap.add_argument("--ablation-candidates", type=int, default=672,
                    help="ablation suite size (7 arms x 6 ops x 16)")
    args = ap.parse_args()

    j = load([Path(r) for r in args.runs])
    if j.empty:
        print("no journal rows")
        return

    j["input_miss"] = (j["input_tokens"] - j["cached_tokens"]).clip(lower=0)
    j["total_out"] = j["output_tokens"]

    print(f"== pilot sample: {len(j)} candidates, "
          f"{j['operator'].nunique()} operators, arms: {sorted(j['arm'].unique())}\n")

    stats = j[["input_tokens", "cached_tokens", "input_miss",
               "output_tokens", "thought_tokens", "gpu_seconds", "wall_time"]]
    print("per-candidate distribution:")
    print(stats.describe(percentiles=[0.5, 0.9]).round(0).to_string(), "\n")

    cache_ratio = j["cached_tokens"].sum() / max(j["input_tokens"].sum(), 1)
    correct_rate = j["correct"].mean()
    print(f"cache hit ratio (input): {cache_ratio*100:.0f}%")
    print(f"correct rate:            {correct_rate*100:.0f}%")
    print(f"error types:             {j[~j['correct']]['error_type'].value_counts().to_dict()}\n")

    per = {"input_miss": j["input_miss"].mean(),
           "cached": j["cached_tokens"].mean(),
           "output": j["output_tokens"].mean()}
    pilot_cost = cost_of(len(j), per)
    print(f"measured pilot cost:        ${pilot_cost:.3f}")
    print(f"avg cost per candidate:     ${pilot_cost/len(j):.4f}")
    print(f"extrapolated main run       ({args.target_candidates} cands): "
          f"${cost_of(args.target_candidates, per):.2f}")
    print(f"extrapolated ablation suite ({args.ablation_candidates} cands): "
          f"${cost_of(args.ablation_candidates, per):.2f}")
    print(f"extrapolated total:         "
          f"${cost_of(args.target_candidates + args.ablation_candidates, per):.2f}")

    # effect snapshot: best speedup per arm/operator at equal budget
    ok = j[j["correct"] == True]  # noqa: E712
    if not ok.empty:
        print("\nbest valid speedup at equal budget (arm x operator):")
        print(ok.groupby(["arm", "operator"])["speedup"].max()
              .unstack(fill_value=0).round(3).to_string())


if __name__ == "__main__":
    main()
