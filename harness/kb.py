"""Success-case knowledge base (AccelOpt-style experience store, concretized).

Build: scan run journals, keep the top correct kernels per operator, persist
code + metadata under kb/. Retrieve: few-shot exemplars for generation prompts
(same operator first, then same category).

  python -m harness.kb --build [--runs 'runs/**'] [--out kb] [--top-n 3]
  python -m harness.kb --show  [--out kb]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KB = ROOT / "kb"


def _code_hash(code: str) -> str:
    return hashlib.sha1(code.strip().encode()).hexdigest()[:12]


def build_kb(runs_glob: str = "runs/**", kb_dir: Path = DEFAULT_KB,
             top_n: int = 3, min_speedup: float = 1.02) -> list[dict]:
    """Scan journals, keep per-operator top-N correct kernels (deduped by code
    hash; below-threshold kept only if the operator has nothing better)."""
    from operators.specs import OPERATORS

    rows = []
    for jf in glob.glob(str(ROOT / runs_glob / "journal.jsonl"), recursive=True):
        run = str(Path(jf).parent.relative_to(ROOT / "runs"))
        for line in open(jf):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("correct") and r.get("code") and r.get("speedup", 0) > 0:
                r["_run"] = run
                rows.append(r)

    by_id = {r["node_id"]: r for r in rows}
    entries = []
    ops = sorted({r["operator"] for r in rows})
    for op in ops:
        cand = [r for r in rows if r["operator"] == op]
        cand.sort(key=lambda r: r["speedup"], reverse=True)
        seen, kept = set(), []
        for r in cand:
            h = _code_hash(r["code"])
            if h in seen:
                continue
            if r["speedup"] < min_speedup and kept:
                continue        # below threshold only acceptable as sole entry
            seen.add(h)
            kept.append((r, h))
            if len(kept) >= top_n:
                break
        for rank, (r, h) in enumerate(kept, 1):
            parent = by_id.get(r.get("parent_id") or "")
            delta = ((r["speedup"] - parent["speedup"]) / parent["speedup"]
                     if parent and parent.get("correct") and parent["speedup"] > 0
                     else None)
            spec = OPERATORS.get(op)
            entries.append({
                "operator": op,
                "category": spec.category if spec else "misc",
                "rank": rank,
                "speedup": round(r["speedup"], 4),
                "bw_utilization": round(r.get("bw_utilization", 0.0), 3),
                "strategy": r.get("strategy", ""),
                "parent_delta": round(delta, 4) if delta is not None else None,
                "source_run": r["_run"],
                "node_id": r["node_id"],
                "code_hash": h,
                "file": f"{op}/{rank}_{r['speedup']:.2f}x.py",
            })

    kb_dir = Path(kb_dir)
    kb_dir.mkdir(parents=True, exist_ok=True)
    code_by_id = {r["node_id"]: r["code"] for r in rows}
    with (kb_dir / "kb.jsonl").open("w") as f:
        for e in entries:
            p = kb_dir / e["file"]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(code_by_id[e["node_id"]])
            f.write(json.dumps(e) + "\n")
    return entries


class KB:
    def __init__(self, entries: list[dict], kb_dir: Path):
        self.entries = entries
        self.kb_dir = Path(kb_dir)

    @classmethod
    def load(cls, kb_dir: Path = DEFAULT_KB) -> "KB":
        kb_dir = Path(kb_dir)
        idx = kb_dir / "kb.jsonl"
        entries = []
        if idx.exists():
            entries = [json.loads(l) for l in idx.read_text().splitlines()
                       if l.strip()]
        return cls(entries, kb_dir)

    def __len__(self):
        return len(self.entries)

    def exemplars_for(self, op_name: str, category: str, k: int = 2,
                      max_lines: int = 80) -> list[tuple[str, dict]]:
        """Same-operator exemplars first, then same-category. Best first."""
        same_op = [e for e in self.entries if e["operator"] == op_name]
        same_cat = [e for e in self.entries
                    if e["operator"] != op_name and e["category"] == category]
        picked = (sorted(same_op, key=lambda e: e["rank"]) +
                  sorted(same_cat, key=lambda e: (e["rank"], -e["speedup"])))[:k]
        out = []
        for e in picked:
            p = self.kb_dir / e["file"]
            if not p.exists():
                continue
            code = "\n".join(p.read_text().splitlines()[:max_lines])
            out.append((code, e))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--runs", default="runs/**")
    ap.add_argument("--out", default=str(DEFAULT_KB))
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--min-speedup", type=float, default=1.02)
    args = ap.parse_args()

    if args.build:
        entries = build_kb(args.runs, Path(args.out), args.top_n, args.min_speedup)
        print(f"KB built: {len(entries)} entries -> {args.out}/kb.jsonl")
    kb = KB.load(Path(args.out))
    if args.show or args.build:
        for e in kb.entries:
            d = f"{e['parent_delta']*100:+.1f}%" if e["parent_delta"] is not None else "  -"
            print(f"  {e['operator']:28s} #{e['rank']} {e['speedup']:7.3f}x "
                  f"bw={e['bw_utilization']*100:3.0f}% strat={e['strategy'][:18]:18s} "
                  f"delta={d:>7s} src={e['source_run']}")


if __name__ == "__main__":
    main()
