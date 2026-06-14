"""GPU worker pool: runs eval_one in subprocesses, one candidate per GPU at a time."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from harness.eval_one import MARKER
from harness.models import EvalResult

ROOT = Path(__file__).resolve().parent.parent


class GPUPool:
    def __init__(self, gpu_ids: list[int], timeout_s: float = 120.0,
                 peak_gbps: float = 1790.0, seed: int = 42,
                 candidates_dir: Path | None = None,
                 launch_stagger_s: float = 0.75):
        self.queue: asyncio.Queue[int] = asyncio.Queue()
        for g in gpu_ids:
            self.queue.put_nowait(g)
        self.timeout_s = timeout_s
        self.peak_gbps = peak_gbps
        self.seed = seed
        self.candidates_dir = candidates_dir
        # Stagger subprocess spawns so we never fire N simultaneous CUDA-context
        # inits + Triton JIT compiles at once (thundering herd on the driver and
        # on the shared ~/.triton cache). Only the *launch moment* is spaced out;
        # the evals still run concurrently, one per GPU.
        self.launch_stagger_s = max(0.0, launch_stagger_s)
        self._spawn_lock = asyncio.Lock()
        self._last_spawn = 0.0

    async def _await_spawn_slot(self):
        """Globally rate-limit process spawns to one per launch_stagger_s."""
        if self.launch_stagger_s <= 0:
            return
        async with self._spawn_lock:
            wait = self.launch_stagger_s - (time.time() - self._last_spawn)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_spawn = time.time()

    async def evaluate(self, operator: str, code: str, tag: str = "cand") -> EvalResult:
        gpu = await self.queue.get()
        try:
            return await self._run_on_gpu(operator, code, gpu, tag)
        finally:
            self.queue.put_nowait(gpu)

    async def _run_on_gpu(self, operator: str, code: str, gpu: int, tag: str) -> EvalResult:
        if self.candidates_dir:
            self.candidates_dir.mkdir(parents=True, exist_ok=True)
            path = self.candidates_dir / f"{tag}.py"
            path.write_text(code)
            cleanup = False
        else:
            fd, p = tempfile.mkstemp(suffix=".py", prefix=f"{operator}_")
            os.write(fd, code.encode())
            os.close(fd)
            path = Path(p)
            cleanup = True

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")

        await self._await_spawn_slot()
        t0 = time.time()
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "harness.eval_one",
                "--operator", operator, "--candidate", str(path),
                "--seed", str(self.seed), "--peak-gbps", str(self.peak_gbps),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                cwd=str(ROOT), env=env)
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self.timeout_s)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return EvalResult(ok=False, error_type="timeout",
                                  error_msg=f"exceeded {self.timeout_s}s",
                                  gpu_seconds=time.time() - t0)
        finally:
            if cleanup:
                path.unlink(missing_ok=True)

        wall = time.time() - t0
        for line in reversed(stdout.decode(errors="replace").splitlines()):
            if line.startswith(MARKER):
                d = json.loads(line[len(MARKER):])
                d["gpu_seconds"] = max(d.get("gpu_seconds", 0.0), wall)
                return EvalResult(ok=True, **{k: d[k] for k in d
                                              if k in EvalResult.__dataclass_fields__})
        return EvalResult(
            ok=False, error_type="crash",
            error_msg=(stderr.decode(errors="replace")[-1500:] or "no RESULT_JSON in output"),
            gpu_seconds=wall)


SELF_TEST_CANDIDATE = '''
import torch
import triton
import triton.language as tl

@triton.jit
def _add(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(o_ptr + offs, x + y, mask=mask)

def triton_run(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, 1024),)
    _add[grid](x, y, out, n, BLOCK=1024, num_warps=4)
    return out
'''


async def _self_test():
    pool = GPUPool([0], timeout_s=180.0)
    res = await pool.evaluate("vector_add", SELF_TEST_CANDIDATE, tag="selftest")
    print(json.dumps(res.__dict__, indent=2))
    assert res.ok and res.correct, "self-test failed"
    print(f"OK  speedup={res.speedup:.3f}x  achieved={res.achieved_gbps:.0f} GB/s "
          f"({res.bw_utilization * 100:.0f}% of peak)")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        asyncio.run(_self_test())
