"""Task card for agent mode: one command prints everything an agent needs.

  python -m harness.task <operator> [--gpu N] [--init]

--init scaffolds the workspace (runs/agent_<op>/attempts/ + STATE.md from the
template below). Idempotent: never overwrites an existing STATE.md or journal.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

STATE_TEMPLATE = """\
# STATE — {op}
status: in_progress
# best / budget 不在這裡手寫(會 drift):看 `python -m harness.task {op}` 任務卡,
# 由 journal 自動算。best.py 用 `python -m harness.task {op} --export-best` 還原。

## tried(每行一筆,新的加在最上面;格式: k<N> | <strategy> | <speedup 或 error 根因>)
(none yet)

## pitfalls(本 op 踩過的環境/API 坑,下一個 agent 不要再踩)
- Triton 3.6 沒有 tl.tanh / tl.math.tanh — 用 `from triton.language.extra
  import libdevice; libdevice.tanh(x)` 或 2*tl.sigmoid(2*x)-1

## next(給下一個 agent 的建議,按優先序)
1. 讀任務卡與 KB 範例,寫第一版 seed kernel
"""


def init_workspace(name: str) -> None:
    ws = ROOT / "runs" / f"agent_{name}"
    (ws / "attempts").mkdir(parents=True, exist_ok=True)
    state = ws / "STATE.md"
    if state.exists():
        print(f"[init] STATE.md already exists — not touching it "
              f"({ws.relative_to(ROOT)}/STATE.md)")
    else:
        state.write_text(STATE_TEMPLATE.format(op=name))
        print(f"[init] created {ws.relative_to(ROOT)}/STATE.md and attempts/")
    journal = ws / "journal.jsonl"
    if journal.exists():
        n = sum(1 for l in journal.read_text().splitlines() if l.strip())
        print(f"[init] journal already has {n} evals — continuing, not resetting")


def export_best(name: str) -> None:
    """Write best.py from the best correct row in the journal (no manual copy)."""
    ws = ROOT / "runs" / f"agent_{name}"
    journal = ws / "journal.jsonl"
    if not journal.exists():
        print(f"[export-best] no journal at {journal.relative_to(ROOT)} — nothing to export")
        return
    rows = [json.loads(l) for l in journal.read_text().splitlines() if l.strip()]
    ok = [r for r in rows if r.get("correct") and r.get("code")]
    if not ok:
        print("[export-best] no correct candidate with code in journal — best.py not written")
        return
    best = max(ok, key=lambda r: r.get("speedup", 0.0))
    (ws / "best.py").write_text(best["code"])
    wc = best.get("winning_config") or {}
    print(f"[export-best] wrote {ws.relative_to(ROOT)}/best.py "
          f"= {best['speedup']:.3f}x (eval #{best['eval_index']})"
          + (f", winning_config={wc}" if wc else ""))


# refs that are already a single fused torch op -> near roofline, tiny budget
FUSED_REF = {"vector_add", "vector_exp", "softmax", "sum", "layer_norm"}


def suggested_budget(spec) -> int:
    if spec.compute_bound:
        return 28          # real optimization ladder (tiling/swizzle/split-K)
    if spec.name in FUSED_REF:
        return 6           # confirm saturation only
    return 12              # multi-pass ref: fusion win + a bit of tuning


def _best_from_runs(op: str):
    """Best correct result for op across ALL runs (api loops + agent)."""
    import glob
    best = None
    for jf in glob.glob(str(ROOT / "runs" / "**" / "journal.jsonl"), recursive=True):
        for line in open(jf):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("operator") == op and r.get("correct"):
                if best is None or r["speedup"] > best["speedup"]:
                    best = r
    return best


def triage():
    from operators.specs import OPERATORS
    print(f"{'operator':28s} {'cat':13s} {'bound':7s} {'budget':6s} "
          f"{'best':>7s} {'bw%':>4s} {'agent':>11s}  recommendation")
    print("-" * 105)
    for name, spec in OPERATORS.items():
        bound = "comp" if spec.compute_bound else "mem"
        budget = suggested_budget(spec)
        best = _best_from_runs(name)
        bs = f"{best['speedup']:.2f}x" if best else "-"
        bw = f"{best['bw_utilization']*100:.0f}" if best else "-"

        ws = ROOT / "runs" / f"agent_{name}"
        jl = ws / "journal.jsonl"
        used = sum(1 for l in jl.read_text().splitlines() if l.strip()) \
            if jl.exists() else 0
        state = ws / "STATE.md"
        st = ""
        if state.exists():
            for line in state.read_text().splitlines():
                if line.startswith("status:"):
                    st = line.split(":", 1)[1].strip().split()[0]
                    break
        agent = f"{used} ev,{st}" if (used or st) else "-"

        saturated = (best is not None and not spec.compute_bound
                     and best["bw_utilization"] >= 0.85) or st == "saturated"
        if saturated:
            rec = "saturated — stop"
        elif best is None:
            rec = ("deep-dive (agent session)" if spec.compute_bound
                   else "not started")
        elif spec.compute_bound:
            rec = "keep climbing (ladder op)"
        elif best["bw_utilization"] >= 0.75:
            rec = "near roofline — few evals left"
        else:
            rec = "room left — continue"
        print(f"{name:28s} {spec.category:13s} {bound:7s} {budget:<6d} "
              f"{bs:>7s} {bw:>4s} {agent:>11s}  {rec}")


def main():
    if "--triage" in sys.argv or "--list" in sys.argv:
        triage()
        return
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        from operators.specs import OPERATORS
        print("usage: python -m harness.task <operator> [--init] [--export-best] [--gpu N]")
        print("       python -m harness.task --triage")
        print("operators:", ", ".join(OPERATORS))
        return
    name = args[0]
    gpu = "<gpu>"
    if "--gpu" in sys.argv:
        gpu = sys.argv[sys.argv.index("--gpu") + 1]

    from operators.specs import OPERATORS, ref_source
    if name not in OPERATORS:
        sys.exit(f"unknown operator: {name} (known: {', '.join(OPERATORS)})")
    spec = OPERATORS[name]

    if "--init" in sys.argv:
        init_workspace(name)
    if "--export-best" in sys.argv:
        export_best(name)

    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    hw = cfg["hardware"]

    print(f"# TASK CARD — {spec.name}")
    print(f"category: {spec.category} "
          f"({'compute-bound' if spec.compute_bound else 'memory-bound'})")
    print(f"hardware: {hw['description']}")
    print(f"tolerance: rtol={spec.rtol}, atol={spec.atol}")
    print(f"\nSIGNATURE:\n  {spec.signature_doc}")
    print("\nPYTORCH REFERENCE:")
    for line in ref_source(spec).strip().splitlines():
        print("  " + line)
    if spec.forbidden_substrings:
        print("\nFORBIDDEN (regex, checked on comment-stripped code):")
        for p in spec.forbidden_substrings:
            print(f"  {p}")
    if not spec.compute_bound:
        print(f"\nROOFLINE: memory-bound; ~85% of {hw['peak_gbps']} GB/s "
              "peak bandwidth = practical optimum (stop there).")

    # KB exemplars
    try:
        from harness.kb import KB
        kb = KB.load()
        ex = kb.exemplars_for(spec.name, spec.category, k=3)
        if ex:
            print("\nKB EXEMPLARS (successful kernels — read them):")
            for _, m in ex:
                print(f"  kb/{m['file']}  {m['speedup']:.2f}x "
                      f"bw={m['bw_utilization']*100:.0f}%  (op={m['operator']})")
        else:
            print("\nKB EXEMPLARS: none for this operator/category yet.")
    except Exception:
        print("\nKB: not built (run: python -m harness.kb --build)")

    # workspace status
    ws = ROOT / "runs" / f"agent_{name}"
    journal = ws / "journal.jsonl"
    print(f"\nWORKSPACE: {ws.relative_to(ROOT)}/")
    if journal.exists():
        rows = [json.loads(l) for l in journal.read_text().splitlines() if l.strip()]
        ok = [r for r in rows if r.get("correct")]
        best = max(ok, key=lambda r: r["speedup"]) if ok else None
        print(f"  budget used: {len(rows)} evals; correct: {len(ok)}")
        if best is not None:
            print(f"  best so far: {best['speedup']:.3f}x "
                  f"(bw={best['bw_utilization']*100:.0f}%, eval #{best['eval_index']})")
            wc = best.get("winning_config") or {}
            if wc:
                print(f"  winning grid config: {wc}  "
                      "(narrow your next grid around this)")
        state = ws / "STATE.md"
        print(f"  STATE.md: {'EXISTS — READ IT FIRST' if state.exists() else 'missing (previous session broke contract)'}")
    else:
        print("  fresh — no previous attempts. Run: "
              f"python -m harness.task {name} --init")

    print("\nEVAL COMMAND (the only way to measure; each call consumes budget):")
    print(f"  CUDA_VISIBLE_DEVICES={gpu} python -m harness.eval_one \\")
    print(f"      --operator {name} --candidate runs/agent_{name}/attempts/k<N>.py \\")
    print(f"      --journal runs/agent_{name}/journal.jsonl --arm agent_{name}")
    print("\nFree pre-checks (no budget cost): python -m py_compile <file>; "
          "API probes like python -c \"import triton.language as tl; "
          "print(hasattr(tl,'tanh'))\"")


if __name__ == "__main__":
    main()
