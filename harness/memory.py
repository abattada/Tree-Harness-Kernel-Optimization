"""Journal (append-only score records) and derived strategy memory."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


class Journal:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self.rows.append(json.loads(line))

    def append(self, row: dict):
        self.rows.append(row)
        with self.path.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")


class MetricsLog:
    """Round-level records: the raw data for budget curves."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(
                "arm,operator,round,candidates_so_far,best_speedup_so_far,"
                "n_correct_so_far,cum_gpu_seconds,cum_tokens\n")

    def append(self, arm, operator, rnd, cands, best, n_correct, gpu_s, tokens):
        with self.path.open("a") as f:
            f.write(f"{arm},{operator},{rnd},{cands},{best:.6f},"
                    f"{n_correct},{gpu_s:.1f},{tokens}\n")


class StrategyMemory:
    """AccelOpt-style experience store, derived from journal rows of this run.

    Aggregates (strategy x op-category) -> improvement stats and renders the
    top entries as prompt lines for the scorer.
    """

    def __init__(self, op_categories: dict[str, str]):
        self.op_categories = op_categories          # operator name -> category
        self.stats = defaultdict(lambda: {"n": 0, "sum": 0.0, "best": 0.0,
                                          "fail": 0})
        self.parent_speedup: dict[str, float] = {}  # node_id -> speedup

    def observe(self, row: dict):
        cat = self.op_categories.get(row["operator"], "misc")
        sid = row["strategy"]
        if sid in ("seed", "sequential_refine"):
            if row["correct"]:
                self.parent_speedup[row["node_id"]] = row["speedup"]
            return
        key = (cat, sid)
        st = self.stats[key]
        if not row["correct"]:
            st["fail"] += 1
            return
        self.parent_speedup[row["node_id"]] = row["speedup"]
        parent = self.parent_speedup.get(row.get("parent_id") or "", None)
        if parent and parent > 0:
            delta = (row["speedup"] - parent) / parent
        else:
            delta = row["speedup"] - 1.0
        st["n"] += 1
        st["sum"] += delta
        st["best"] = max(st["best"], delta)

    def relative_reward(self, row: dict) -> float:
        """Reward for the bandit: clipped relative improvement vs parent."""
        if not row["correct"]:
            return 0.0
        parent = self.parent_speedup.get(row.get("parent_id") or "", None)
        base = parent if parent and parent > 0 else 1.0
        return max(0.0, min(1.0, (row["speedup"] - base) / base))

    def lines_for(self, op_category: str, top_m: int = 6) -> list[str]:
        entries = []
        for (cat, sid), st in self.stats.items():
            if cat != op_category:
                continue
            n = st["n"]
            mean = st["sum"] / n if n else 0.0
            entries.append((st["best"], f"- {sid}: tried {n + st['fail']}x in "
                            f"category {cat}; mean delta {mean*100:+.1f}%, "
                            f"best {st['best']*100:+.1f}%, {st['fail']} failures"))
        entries.sort(reverse=True, key=lambda t: t[0])
        return [e[1] for e in entries[:top_m]]
