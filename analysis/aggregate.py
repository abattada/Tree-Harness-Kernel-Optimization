#!/usr/bin/env python
"""Aggregate journals from multiple arms into comparison artifacts.

Everything is derived from journal.jsonl / metrics.csv — no hand-entered
numbers. Outputs: compare_table.md, budget_curves.csv (+png), strategy_report.csv,
fast_p.csv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

FAST_P = [1.0, 1.2, 1.5]


def load_runs(run_dirs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    jr, mr = [], []
    for d in run_dirs:
        jpath, mpath = d / "journal.jsonl", d / "metrics.csv"
        if jpath.exists():
            rows = [json.loads(l) for l in jpath.read_text().splitlines() if l.strip()]
            for r in rows:
                r.pop("code", None)
            jr.extend(rows)
        if mpath.exists():
            mr.append(pd.read_csv(mpath))
    journal = pd.DataFrame(jr)
    metrics = pd.concat(mr, ignore_index=True) if mr else pd.DataFrame()
    return journal, metrics


def best_table(j: pd.DataFrame) -> pd.DataFrame:
    ok = j[j["correct"] == True]  # noqa: E712
    best = (ok.groupby(["arm", "operator"])["speedup"].max()
            .unstack(fill_value=0.0).round(3))
    return best


def fast_p_table(j: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm, g in j.groupby("arm"):
        ops = g["operator"].unique()
        ok = g[g["correct"] == True]  # noqa: E712
        per_op_best = ok.groupby("operator")["speedup"].max()
        row = {"arm": arm, "n_operators": len(ops)}
        for p in FAST_P:
            row[f"fast_{p}"] = round(
                sum(per_op_best.get(o, 0.0) > p for o in ops) / max(len(ops), 1), 3)
        row["n_candidates"] = len(g)
        row["correct_rate"] = round(len(ok) / max(len(g), 1), 3)
        row["gpu_minutes"] = round(g["gpu_seconds"].sum() / 60, 1)
        row["mtokens"] = round((g["input_tokens"].sum() + g["output_tokens"].sum()) / 1e6, 2)
        rows.append(row)
    return pd.DataFrame(rows)


def budget_curves(j: pd.DataFrame) -> pd.DataFrame:
    """best-so-far speedup vs eval_index, per (arm, operator)."""
    out = []
    for (arm, op), g in j.groupby(["arm", "operator"]):
        g = g.sort_values("eval_index")
        best = 0.0
        for _, r in g.iterrows():
            if r["correct"] and r["speedup"] > best:
                best = r["speedup"]
            out.append({"arm": arm, "operator": op,
                        "eval_index": int(r["eval_index"]) + 1,
                        "best_speedup_so_far": round(best, 4)})
    return pd.DataFrame(out)


def strategy_report(j: pd.DataFrame) -> pd.DataFrame:
    sp = j.set_index("node_id")["speedup"].to_dict()
    corr = j.set_index("node_id")["correct"].to_dict()
    rows = []
    for _, r in j.iterrows():
        if r["strategy"] in ("seed", "sequential_refine"):
            continue
        parent = r.get("parent_id")
        base = sp.get(parent, 1.0) if parent and corr.get(parent) else 1.0
        delta = (r["speedup"] - base) / base if r["correct"] and base > 0 else None
        rows.append({"arm": r["arm"], "strategy": r["strategy"],
                     "operator": r["operator"], "correct": r["correct"],
                     "rel_improvement": delta})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    agg = (df.groupby(["arm", "strategy"])
           .agg(n=("correct", "size"),
                correct_rate=("correct", "mean"),
                mean_delta=("rel_improvement", "mean"),
                best_delta=("rel_improvement", "max"))
           .reset_index().round(4))
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    j, m = load_runs([Path(r) for r in args.runs])
    if j.empty:
        print("no journal rows found")
        return

    best = best_table(j)
    fp = fast_p_table(j)
    curves = budget_curves(j)
    strat = strategy_report(j)

    curves.to_csv(out / "budget_curves.csv", index=False)
    strat.to_csv(out / "strategy_report.csv", index=False)
    fp.to_csv(out / "fast_p.csv", index=False)

    md = ["# Ablation comparison", "", "## Best valid speedup (arm x operator)", "",
          best.to_markdown(), "", "## fast_p / cost per arm", "",
          fp.to_markdown(index=False), ""]
    (out / "compare_table.md").write_text("\n".join(md))
    print((out / "compare_table.md").read_text())

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ops = curves["operator"].unique()
        fig, axes = plt.subplots(1, len(ops), figsize=(5 * len(ops), 4),
                                 squeeze=False)
        for ax, op in zip(axes[0], ops):
            for arm, g in curves[curves["operator"] == op].groupby("arm"):
                ax.plot(g["eval_index"], g["best_speedup_so_far"],
                        marker=".", label=arm)
            ax.set_title(op)
            ax.set_xlabel("candidates evaluated")
            ax.set_ylabel("best valid speedup")
            ax.axhline(1.0, ls="--", c="gray", lw=0.8)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "budget_curves.png", dpi=140)
        print(f"wrote {out / 'budget_curves.png'}")
    except Exception as e:
        print(f"[warn] plot skipped: {e}")


if __name__ == "__main__":
    main()
