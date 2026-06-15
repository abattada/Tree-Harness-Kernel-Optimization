#!/usr/bin/env python
"""Generate report data figures from run journals (no hand-entered numbers).

Outputs to report/figures/:
  deepseek_speedup.png   per-operator best speedup (DeepSeek API loop), roofline line
  int4gemm_ladder.png    Claude Code agent best-so-far speedup over evals (int4_gemm)
  budget_curves.png      best-so-far vs candidates for the DeepSeek focus operators
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)

DEEPSEEK_RUNS = [
    ROOT / "runs/minimal_kb_0614_091650",   # focus (84 evals, 7 memory-bound)
    ROOT / "runs/minimal_kb_0614_093601",   # control (~1.0x ops)
    ROOT / "runs/minimal_kb_0614_120740",   # compute-bound attempts
]
FOCUS_RUN = ROOT / "runs/minimal_kb_0614_091650"
AGENT_RUN = ROOT / "runs/agent_int4_gemm"

# operators we treat as the "control / no-headroom" group (honesty bar)
CONTROL = {"vector_add", "vector_exp", "sum", "softmax", "layer_norm"}


def load(run: Path) -> list[dict]:
    p = run / "journal.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def best_per_op(runs) -> dict[str, float]:
    best: dict[str, float] = {}
    for run in runs:
        for r in load(run):
            if r.get("correct") and r.get("speedup", 0) > 0:
                best[r["operator"]] = max(best.get(r["operator"], 0.0), r["speedup"])
    return best


# ---------------------------------------------------------- Fig: DeepSeek bars
def fig_deepseek_speedup():
    best = best_per_op(DEEPSEEK_RUNS)
    # order: focus (desc) then control (desc), drop compute-bound 0s
    items = [(k, v) for k, v in best.items() if v > 0]
    focus = sorted([kv for kv in items if kv[0] not in CONTROL],
                   key=lambda kv: kv[1], reverse=True)
    ctrl = sorted([kv for kv in items if kv[0] in CONTROL],
                  key=lambda kv: kv[1], reverse=True)
    ordered = focus + ctrl
    names = [k for k, _ in ordered]
    vals = [v for _, v in ordered]
    colors = ["#2c7fb8" if n not in CONTROL else "#bdbdbd" for n in names]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    bars = ax.bar(names, vals, color=colors)
    ax.axhline(1.0, ls="--", c="gray", lw=1, label="PyTorch eager (1.0x)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("best valid speedup (x)")
    ax.set_title("DeepSeek API loop — best speedup per operator "
                 "(blue = headroom group, grey = control)")
    ax.set_ylim(0, max(vals) * 1.18)
    plt.xticks(rotation=35, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "deepseek_speedup.png", dpi=150)
    plt.close(fig)
    print("wrote", FIG / "deepseek_speedup.png")


# ------------------------------------------------------ Fig: int4_gemm ladder
def fig_int4gemm_ladder():
    rows = sorted(load(AGENT_RUN), key=lambda r: r.get("eval_index", 0))
    xs, ys, best = [], [], 0.0
    pts_ok = []
    for r in rows:
        i = int(r.get("eval_index", 0))
        if r.get("correct") and r.get("speedup", 0) > best:
            best = r["speedup"]
        xs.append(i)
        ys.append(best)
        if r.get("correct"):
            pts_ok.append((i, r["speedup"]))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot([x + 1 for x in xs], ys, marker="o", color="#d95f02",
            label="best-so-far")
    if pts_ok:
        ax.scatter([x + 1 for x, _ in pts_ok], [y for _, y in pts_ok],
                   color="#1b9e77", s=22, zorder=3, label="correct candidate")
    ax.axhline(1.0, ls="--", c="gray", lw=1)
    ax.set_xlabel("candidates evaluated")
    ax.set_ylabel("speedup vs cuBLAS+dequant (x)")
    ax.set_title("Claude Code agent — int4_gemm optimization ladder "
                 f"(best {best:.3f}x)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "int4gemm_ladder.png", dpi=150)
    plt.close(fig)
    print("wrote", FIG / "int4gemm_ladder.png")


# ------------------------------------------------------ Fig: budget curves
def fig_budget_curves():
    rows = load(FOCUS_RUN)
    by_op: dict[str, list] = {}
    for r in rows:
        by_op.setdefault(r["operator"], []).append(r)
    ops = [o for o in by_op if o not in CONTROL]
    ops.sort()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for op in ops:
        g = sorted(by_op[op], key=lambda r: r.get("eval_index", 0))
        xs, ys, best, k = [], [], 0.0, 0
        for r in g:
            k += 1
            if r.get("correct") and r.get("speedup", 0) > best:
                best = r["speedup"]
            xs.append(k)
            ys.append(best)
        # keep only operators that visibly improve AFTER their first correct
        # kernel (>=3% refinement gain); flat-from-first-success operators add
        # no "optimization progress" signal.
        first_ok = next((v for v in ys if v > 0), 0.0)
        if first_ok <= 0 or max(ys) <= first_ok * 1.03:
            continue
        ax.plot(xs, ys, marker="o", ms=4, label=op)
    ax.axhline(1.0, ls="--", c="gray", lw=0.8)
    ax.set_xlabel("candidates evaluated (per operator)")
    ax.set_ylabel("best valid speedup (x)")
    ax.set_title("DeepSeek API loop — operators that improve over iterations")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "budget_curves.png", dpi=150)
    plt.close(fig)
    print("wrote", FIG / "budget_curves.png")


if __name__ == "__main__":
    fig_deepseek_speedup()
    fig_int4gemm_ladder()
    fig_budget_curves()
