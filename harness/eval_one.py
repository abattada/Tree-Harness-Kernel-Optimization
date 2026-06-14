"""Evaluate one candidate kernel in an isolated process.

Run as:  python -m harness.eval_one --operator NAME --candidate FILE --seed 42
Prints exactly one line starting with RESULT_JSON: followed by the result dict.
All timing/correctness code lives here — candidate code never touches the clock.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import traceback


MARKER = "RESULT_JSON:"

# agent-mode journaling state (set in main; emit() runs exactly once/process)
_J = {"path": None, "arm": "agent", "operator": "", "code": "", "strategy": "agent"}


def emit(d: dict):
    print(MARKER + json.dumps(d), flush=True)
    if _J["path"]:
        _append_journal(d)


def _append_journal(d: dict):
    import os
    import uuid
    path = _J["path"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        with open(path) as f:
            idx = sum(1 for line in f if line.strip())
    except OSError:
        idx = 0
    ok = bool(d.get("correct"))
    row = {
        "node_id": uuid.uuid4().hex[:12], "arm": _J["arm"],
        "operator": _J["operator"], "round": idx, "eval_index": idx,
        "parent_id": None, "strategy": _J["strategy"],
        "correct": ok, "error_type": d.get("error_type", ""),
        "error_msg": (d.get("error_msg", "") or "")[-500:],
        "pytorch_ms": d.get("pytorch_ms", 0.0), "triton_ms": d.get("triton_ms", 0.0),
        "speedup": d.get("speedup", 0.0), "max_abs_err": d.get("max_abs_err", 0.0),
        "achieved_gbps": d.get("achieved_gbps", 0.0),
        "bw_utilization": d.get("bw_utilization", 0.0),
        "score_speedup": d.get("speedup", 0.0) if ok else 0.0,
        "score_headroom": 0.0, "score_confidence": 0.0,
        "score_final": d.get("speedup", 0.0) if ok else -1.0,
        "selected_into_beam": False, "llm_confidence": -1.0,
        "logprob_conf": None, "headroom_pct": 0.0, "bottleneck": "",
        "suggested_strategies": [], "assess_confidence": -1,
        "gpu_seconds": d.get("gpu_seconds", 0.0),
        "winning_config": d.get("winning_config", {}),
        "input_tokens": 0, "output_tokens": 0, "cached_tokens": 0,
        "thought_tokens": 0, "wall_time": d.get("gpu_seconds", 0.0),
        "code": _J["code"],
    }
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def _config_to_dict(cfg) -> dict:
    """Serialize a triton.Config (or fall back to str). Best-effort."""
    try:
        d = dict(getattr(cfg, "kwargs", {}) or {})
        for attr in ("num_warps", "num_stages", "num_ctas"):
            v = getattr(cfg, attr, None)
            if v is not None:
                d[attr] = v
        return d if d else {"repr": str(cfg)}
    except Exception:
        return {"repr": str(cfg)}


def _collect_winning_config(mod) -> dict:
    """Scan candidate module globals for triton Autotuner instances and read the
    config each one selected. Duck-typed so we don't hard-bind the triton path;
    best-effort — autotuners hidden inside functions are simply not seen."""
    out = {}
    try:
        for name, obj in vars(mod).items():
            if hasattr(obj, "best_config") and hasattr(obj, "cache"):
                best = getattr(obj, "best_config", None)
                if best is None:
                    cache = getattr(obj, "cache", {}) or {}
                    if len(cache) == 1:
                        best = next(iter(cache.values()))
                if best is not None:
                    out[name] = _config_to_dict(best)
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--operator", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--peak-gbps", type=float, default=1790.0)
    ap.add_argument("--journal", default=None,
                    help="append an aggregate.py-compatible row to this jsonl "
                         "(agent mode; eval_index = current line count)")
    ap.add_argument("--arm", default="agent")
    ap.add_argument("--strategy", default="agent",
                    help="direction label recorded in the journal row "
                         "(agent mode; e.g. autotune_grid)")
    args = ap.parse_args()
    _J["path"] = args.journal
    _J["arm"] = args.arm
    _J["operator"] = args.operator
    _J["strategy"] = args.strategy

    # When several eval_one processes are fired at once (e.g. agent mode running
    # one per GPU with `&`), de-sync their CUDA-init / Triton-compile bursts.
    # Off by default (single launches unaffected); export EVAL_STARTUP_JITTER_S=2
    # before a parallel batch. Harmless to timing — do_bench measures internally.
    import os
    import random
    jitter = float(os.environ.get("EVAL_STARTUP_JITTER_S", "0") or 0)
    if jitter > 0:
        time.sleep(random.uniform(0, jitter))

    t0 = time.time()
    out = {
        "correct": False, "error_type": "", "error_msg": "",
        "pytorch_ms": 0.0, "triton_ms": 0.0, "speedup": 0.0,
        "max_abs_err": 0.0, "mean_abs_err": 0.0,
        "achieved_gbps": 0.0, "bw_utilization": 0.0, "gpu_seconds": 0.0,
        "winning_config": {},
    }

    try:
        import torch
        import triton.testing as tt
        from operators.specs import get_spec
    except Exception:
        out["error_type"] = "harness"
        out["error_msg"] = traceback.format_exc()[-1500:]
        emit(out)
        return

    spec = get_spec(args.operator)

    # naive anti-cheat: reject candidates calling the reference op directly
    # (entries are regex patterns; e.g. (?<!tl)\.sum\( hits x.sum( but not tl.sum()
    # Comments and docstrings are stripped first — mentioning F.gelu in a
    # comment is not cheating.
    import re
    try:
        src = open(args.candidate).read()
        _J["code"] = src
        code_only = re.sub(r'"""(?:.|\n)*?"""|\'\'\'(?:.|\n)*?\'\'\'', "", src)
        code_only = re.sub(r"#[^\n]*", "", code_only)
        hits = [p for p in spec.forbidden_substrings if re.search(p, code_only)]
        if hits:
            out["error_type"] = "cheat"
            out["error_msg"] = f"forbidden call(s) in candidate: {hits}"
            emit(out)
            return
    except OSError:
        out["error_type"] = "harness"
        out["error_msg"] = "cannot read candidate file"
        emit(out)
        return

    # import candidate (Triton JIT compilation happens on first call)
    try:
        mod_spec = importlib.util.spec_from_file_location("candidate", args.candidate)
        mod = importlib.util.module_from_spec(mod_spec)
        sys.modules["candidate"] = mod
        mod_spec.loader.exec_module(mod)
        fn = getattr(mod, "triton_run")
    except Exception:
        out["error_type"] = "compile"
        out["error_msg"] = traceback.format_exc()[-1500:]
        emit(out)
        return

    try:
        device = "cuda"
        torch.manual_seed(args.seed)
        inputs = spec.make_inputs(args.seed, device)
        ref_out = spec.ref(*inputs)
        torch.cuda.synchronize()
    except Exception:
        out["error_type"] = "harness"
        out["error_msg"] = traceback.format_exc()[-1500:]
        emit(out)
        return

    # first candidate call: compile + correctness
    try:
        cand_out = fn(*inputs)
        torch.cuda.synchronize()
        if not torch.is_tensor(cand_out):
            raise TypeError(f"triton_run returned {type(cand_out)}, expected Tensor")
        if cand_out.shape != ref_out.shape:
            raise ValueError(f"shape {tuple(cand_out.shape)} != ref {tuple(ref_out.shape)}")
        diff = (cand_out.float() - ref_out.float()).abs()
        out["max_abs_err"] = float(diff.max())
        out["mean_abs_err"] = float(diff.mean())
        out["correct"] = bool(torch.allclose(
            cand_out.float(), ref_out.float(), rtol=spec.rtol, atol=spec.atol))
        if not out["correct"]:
            out["error_type"] = "wrong_output"
            out["error_msg"] = (f"max_abs_err={out['max_abs_err']:.6g} "
                                f"mean_abs_err={out['mean_abs_err']:.6g} "
                                f"(rtol={spec.rtol}, atol={spec.atol})")
    except Exception:
        out["error_type"] = "runtime"
        out["error_msg"] = traceback.format_exc()[-1500:]
        emit(out)
        return

    # benchmark both sides identically (median over reps, includes output alloc on both)
    try:
        out["pytorch_ms"] = float(tt.do_bench(
            lambda: spec.ref(*inputs), warmup=25, rep=100, return_mode="median"))
        out["triton_ms"] = float(tt.do_bench(
            lambda: fn(*inputs), warmup=25, rep=100, return_mode="median"))
        out["speedup"] = out["pytorch_ms"] / out["triton_ms"] if out["triton_ms"] > 0 else 0.0
        nbytes = spec.bytes_moved(inputs, ref_out)
        out["achieved_gbps"] = nbytes / (out["triton_ms"] * 1e-3) / 1e9
        out["bw_utilization"] = out["achieved_gbps"] / args.peak_gbps
    except Exception:
        out["error_type"] = "runtime"
        out["error_msg"] = traceback.format_exc()[-1500:]
        emit(out)
        return
    finally:
        out["gpu_seconds"] = time.time() - t0

    # record which autotune grid config(s) won (best-effort; {} if no autotune)
    out["winning_config"] = _collect_winning_config(mod)

    emit(out)


if __name__ == "__main__":
    main()
