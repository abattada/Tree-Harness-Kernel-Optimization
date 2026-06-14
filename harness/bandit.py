"""Strategy selectors: UCB1 bandit (per operator category) and uniform random."""
from __future__ import annotations

import math
import random
from collections import defaultdict

from harness.strategies import applicable


class BaseSelector:
    def select(self, op_category: str, k: int, exclude: set[str] | None = None) -> list[str]:
        raise NotImplementedError

    def update(self, op_category: str, strategy_id: str, reward: float):
        pass

    def snapshot(self) -> dict:
        return {}


class NoneSelector(BaseSelector):
    """No strategy assignment: tree expansion falls back to free refinement
    (the minimal-arm / tree_free configuration)."""

    def select(self, op_category, k, exclude=None):
        return []


class RandomSelector(BaseSelector):
    """Uniform random over applicable strategies (the 'paper default' baseline)."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def select(self, op_category, k, exclude=None):
        cands = [s.id for s in applicable(op_category) if s.id not in (exclude or set())]
        self.rng.shuffle(cands)
        return cands[:k]


class UCB1Selector(BaseSelector):
    """UCB1 per op category. Reward in [0, 1] = clipped relative improvement."""

    def __init__(self, seed: int = 0, c: float = 1.2):
        self.rng = random.Random(seed)
        self.c = c
        self.n: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.sum_r: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def _ucb(self, cat: str, sid: str, total: int) -> float:
        n = self.n[cat][sid]
        if n == 0:
            return float("inf")
        mean = self.sum_r[cat][sid] / n
        return mean + self.c * math.sqrt(math.log(max(total, 2)) / n)

    def select(self, op_category, k, exclude=None):
        cands = [s.id for s in applicable(op_category) if s.id not in (exclude or set())]
        self.rng.shuffle(cands)            # random tie-break for unexplored arms
        total = sum(self.n[op_category].values()) + 1
        cands.sort(key=lambda sid: self._ucb(op_category, sid, total), reverse=True)
        return cands[:k]

    def update(self, op_category, strategy_id, reward):
        r = max(0.0, min(1.0, reward))
        self.n[op_category][strategy_id] += 1
        self.sum_r[op_category][strategy_id] += r

    def snapshot(self):
        out = {}
        for cat, arms in self.n.items():
            out[cat] = {sid: {"n": n, "mean_reward":
                              (self.sum_r[cat][sid] / n if n else 0.0)}
                        for sid, n in arms.items()}
        return out


def make_selector(kind: str, seed: int) -> BaseSelector:
    if kind == "bandit":
        return UCB1Selector(seed=seed)
    if kind == "random":
        return RandomSelector(seed=seed)
    if kind == "none":
        return NoneSelector()
    raise ValueError(f"unknown selector: {kind}")
