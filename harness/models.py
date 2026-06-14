"""Shared data models."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional

from pydantic import BaseModel, Field


class NodeAssessment(BaseModel):
    """Structured output of the LLM scorer."""
    headroom_pct: float = Field(
        description="Estimated remaining speedup headroom in percent "
                    "(0 = already optimal, 50 = ~1.5x more is plausible).")
    bottleneck: str = Field(
        description="Main bottleneck: memory_bandwidth | compute | launch_overhead "
                    "| occupancy | algorithm | unknown")
    suggested_strategies: list[str] = Field(
        description="Strategy ids from the strategy library worth trying next "
                    "on this kernel, best first.")
    confidence: int = Field(ge=0, le=100,
                            description="Confidence in this assessment, 0-100.")
    reasoning: str = Field(description="One short paragraph justifying the estimate.")


@dataclass
class EvalResult:
    ok: bool = False                  # process ran and produced a result
    correct: bool = False
    error_type: str = ""              # compile | runtime | wrong_output | timeout | cheat | parse
    error_msg: str = ""
    pytorch_ms: float = 0.0
    triton_ms: float = 0.0
    speedup: float = 0.0
    max_abs_err: float = 0.0
    mean_abs_err: float = 0.0
    achieved_gbps: float = 0.0
    bw_utilization: float = 0.0
    gpu_seconds: float = 0.0          # wall time spent in the eval subprocess
    winning_config: dict = field(default_factory=dict)  # autotune grid winner(s), kernel-name -> config


@dataclass
class Node:
    operator: str
    round: int
    strategy: str
    code: str
    parent_id: Optional[str] = None
    node_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    eval_index: int = -1              # global order of evaluation within the run

    # evaluation
    result: Optional[EvalResult] = None

    # llm metadata
    llm_confidence: float = -1.0      # verbalized confidence 0-100 (-1 = absent)
    logprob_conf: Optional[float] = None  # DeepConf-style min-window mean logprob
    gen_notes: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    thought_tokens: int = 0

    # scorer
    headroom_pct: float = 0.0
    bottleneck: str = ""
    suggested_strategies: list = field(default_factory=list)
    assess_confidence: int = -1

    # composite score breakdown
    score_speedup: float = 0.0
    score_headroom: float = 0.0
    score_confidence: float = 0.0
    score_final: float = 0.0
    selected_into_beam: bool = False

    wall_time: float = 0.0

    def journal_row(self, arm: str) -> dict:
        r = self.result or EvalResult()
        return {
            "node_id": self.node_id, "arm": arm, "operator": self.operator,
            "round": self.round, "eval_index": self.eval_index,
            "parent_id": self.parent_id, "strategy": self.strategy,
            "correct": r.correct, "error_type": r.error_type,
            "error_msg": r.error_msg[-500:],
            "pytorch_ms": r.pytorch_ms, "triton_ms": r.triton_ms,
            "speedup": r.speedup, "max_abs_err": r.max_abs_err,
            "achieved_gbps": r.achieved_gbps, "bw_utilization": r.bw_utilization,
            "score_speedup": self.score_speedup,
            "score_headroom": self.score_headroom,
            "score_confidence": self.score_confidence,
            "score_final": self.score_final,
            "selected_into_beam": self.selected_into_beam,
            "llm_confidence": self.llm_confidence,
            "logprob_conf": self.logprob_conf,
            "headroom_pct": self.headroom_pct, "bottleneck": self.bottleneck,
            "suggested_strategies": self.suggested_strategies,
            "assess_confidence": self.assess_confidence,
            "gpu_seconds": r.gpu_seconds,
            "winning_config": r.winning_config,
            "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens, "thought_tokens": self.thought_tokens,
            "wall_time": self.wall_time,
            "code": self.code,
        }


def node_public_dict(n: Node) -> dict:
    d = asdict(n)
    d.pop("code", None)
    return d
