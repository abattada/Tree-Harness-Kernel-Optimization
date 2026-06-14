"""Composite node scoring with roofline anchor. Records the full breakdown."""
from __future__ import annotations

import math

from harness.models import Node


def apply_score(node: Node, cfg: dict, compute_bound: bool):
    """Fill node.score_* fields in place. cfg = config['scorer'] + ['confidence']."""
    r = node.result
    if r is None or not r.correct:
        node.score_speedup = 0.0
        node.score_headroom = 0.0
        node.score_confidence = 0.0
        node.score_final = -1.0
        return

    scorer_on = cfg.get("enabled", True)
    roofline_on = cfg.get("roofline_anchor", True)
    bw_cap = cfg.get("roofline_bw_cap", 0.85)
    beta = cfg.get("headroom_beta", 0.3)
    conf_in_score = cfg.get("conf_in_score", False)
    conf_eps = cfg.get("conf_epsilon", 1e-3)

    node.score_speedup = r.speedup

    headroom = node.headroom_pct / 100.0 if scorer_on else 0.0
    if roofline_on and not compute_bound and r.bw_utilization >= bw_cap:
        headroom = 0.0          # at the memory roofline: no headroom, period
        node.headroom_pct = 0.0
    node.score_headroom = beta * headroom * r.speedup if scorer_on else 0.0

    conf = 0.0
    if conf_in_score:
        if node.logprob_conf is not None:
            conf = math.exp(max(-20.0, node.logprob_conf))     # (0, 1]
        elif node.llm_confidence >= 0:
            conf = node.llm_confidence / 100.0
    node.score_confidence = conf_eps * conf                    # tie-break scale

    node.score_final = node.score_speedup + node.score_headroom + node.score_confidence
