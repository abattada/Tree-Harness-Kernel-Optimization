#!/usr/bin/env python
"""Run multiple ablation arms sequentially (fair GPU timing), then aggregate.

  python run_ablation.py --arms full sequential_baseline \
      --operators vector_add softmax --total-candidates 6
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True,
                    help="arm names matching configs/arms/<name>.yaml")
    ap.add_argument("--operators", nargs="+", default=["all"])
    ap.add_argument("--total-candidates", type=int)
    ap.add_argument("--rounds", type=int)
    ap.add_argument("--gpus", nargs="+",
                    help="GPU ids, space or comma separated: --gpus 3 4 5 / --gpus 3,4,5")
    ap.add_argument("--tag", default=datetime.now().strftime("%m%d_%H%M%S"))
    args = ap.parse_args()
    if args.gpus:
        args.gpus = [str(int(g)) for tok in args.gpus for g in tok.split(",") if g]

    batch_dir = ROOT / "runs" / f"ablation_{args.tag}"
    run_dirs = []
    for arm in args.arms:
        arm_cfg = ROOT / "configs" / "arms" / f"{arm}.yaml"
        if not arm_cfg.exists():
            sys.exit(f"missing config: {arm_cfg}")
        out = batch_dir / arm
        cmd = [sys.executable, str(ROOT / "runner.py"),
               "--config", str(arm_cfg), "--out", str(out),
               "--operators", *args.operators]
        if args.total_candidates:
            cmd += ["--total-candidates", str(args.total_candidates)]
        if args.rounds:
            cmd += ["--rounds", str(args.rounds)]
        if args.gpus:
            cmd += ["--gpus", *map(str, args.gpus)]
        print(f"\n=== ARM {arm} ===\n$ {' '.join(cmd)}")
        rc = subprocess.call(cmd, cwd=str(ROOT))
        if rc != 0:
            print(f"[warn] arm {arm} exited with {rc}; continuing")
        run_dirs.append(out)

    print("\n=== AGGREGATE ===")
    subprocess.call([sys.executable, str(ROOT / "analysis" / "aggregate.py"),
                     "--runs", *map(str, run_dirs),
                     "--out", str(batch_dir / "comparison")], cwd=str(ROOT))


if __name__ == "__main__":
    main()
