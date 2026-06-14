"""DeepSeek API layer (OpenAI-compatible): candidate generation, node
assessment, capability probe.

Run `python -m harness.llm --probe` to verify every call pattern the harness
uses (async generation, thinking parameter, logprobs, JSON mode, automatic
context caching) against the configured model.

Notes (verified against api-docs.deepseek.com, 2026-06):
- models: deepseek-v4-flash, deepseek-v4-pro (1M ctx, 384K max output);
  thinking is ON by default, controlled via `thinking={"type": ..., "reasoning_effort": ...}`.
- logprobs / top_logprobs are native chat-completion parameters.
- context caching is automatic (prefix match); usage reports
  prompt_cache_hit_tokens / prompt_cache_miss_tokens.
- response_format json_object requires the word "JSON" + key description in
  the prompt, otherwise the model may emit whitespace forever.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import re
import sys

import openai
from openai import AsyncOpenAI

from harness.models import NodeAssessment

BASE_URL = "https://api.deepseek.com"
CODE_RE = re.compile(r"```python\s*(.*?)```", re.DOTALL)
CONF_RE = re.compile(r'CONF:\s*(\{.*\})')
JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


class LLMClient:
    def __init__(self, model: str, reasoning_effort: str = "high",
                 max_concurrent: int = 10, max_output_tokens: int = 24576,
                 temperature: float = 1.0):
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("set DEEPSEEK_API_KEY")
        self.client = AsyncOpenAI(api_key=api_key, base_url=BASE_URL,
                                  max_retries=3, timeout=1200.0)
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.sem = asyncio.Semaphore(max_concurrent)
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

        # capabilities discovered by setup()
        self.supports_thinking_param = True
        self.supports_logprobs = False
        self._system_text: str = ""

    # ------------------------------------------------------------- plumbing

    async def _chat(self, messages, thinking: dict | None, retries: int = 4,
                    **kw):
        extra = {}
        if thinking is not None and self.supports_thinking_param:
            extra["thinking"] = thinking
        delay = 2.0
        last = None
        for _ in range(retries + 1):
            try:
                async with self.sem:
                    return await self.client.chat.completions.create(
                        model=self.model, messages=messages,
                        max_tokens=self.max_output_tokens,
                        temperature=self.temperature,
                        extra_body=extra or None, **kw)
            except (openai.RateLimitError, openai.InternalServerError,
                    openai.APIConnectionError, openai.APITimeoutError) as e:
                last = e
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60.0)
        raise last

    def _gen_thinking(self) -> dict:
        return {"type": "enabled", "reasoning_effort": self.reasoning_effort}

    # ------------------------------------------------------------ lifecycle

    async def setup(self, system_text: str) -> dict:
        """Probe capabilities. Context caching is automatic on DeepSeek —
        nothing to create; the stable system prompt is the cached prefix."""
        meta = {"model": self.model, "base_url": BASE_URL}
        self._system_text = system_text

        # thinking parameter support
        try:
            await self._chat([{"role": "user", "content": "Say OK."}],
                             thinking={"type": "disabled"}, retries=1)
            self.supports_thinking_param = True
        except openai.BadRequestError as e:
            self.supports_thinking_param = False
            meta["thinking_error"] = str(e)[:200]
        meta["thinking_param"] = self.supports_thinking_param

        # logprobs support (DeepConf needs this)
        try:
            r = await self._chat([{"role": "user", "content": "Say OK."}],
                                 thinking={"type": "disabled"}, retries=1,
                                 logprobs=True, top_logprobs=3)
            lp = r.choices[0].logprobs
            self.supports_logprobs = bool(lp and lp.content)
        except openai.BadRequestError as e:
            self.supports_logprobs = False
            meta["logprobs_error"] = str(e)[:200]
        meta["logprobs"] = self.supports_logprobs

        # cache plumbing check (fields exist even on a cold call)
        try:
            r = await self._chat(
                [{"role": "system", "content": system_text},
                 {"role": "user", "content": "Reply with the word: ready"}],
                thinking={"type": "disabled"}, retries=1)
            u = r.usage
            meta["cache_fields"] = hasattr(u, "prompt_cache_hit_tokens") or \
                "prompt_cache_hit_tokens" in (getattr(u, "model_extra", {}) or {})
        except Exception as e:
            meta["cache_fields"] = False
            meta["cache_error"] = str(e)[:200]
        return meta

    async def close(self):
        await self.client.close()

    # ----------------------------------------------------------- generation

    async def generate_candidate(self, prompt: str) -> dict:
        """Returns {code, llm_confidence, logprob_conf, notes, usage,
        system, prompt, raw_response, reasoning} or {error, ...}. The last four
        are for transcript logging — always populated when a response came back."""
        kw = {}
        if self.supports_logprobs:
            kw = {"logprobs": True, "top_logprobs": 3}
        try:
            r = await self._chat(
                [{"role": "system", "content": self._system_text},
                 {"role": "user", "content": prompt}],
                thinking=self._gen_thinking(), **kw)
        except Exception as e:
            return {"error": f"api: {e}", "system": self._system_text,
                    "prompt": prompt}

        choice = r.choices[0]
        text = choice.message.content or ""
        tx = {"system": self._system_text, "prompt": prompt,
              "raw_response": text, "reasoning": _reasoning(choice)}
        m = CODE_RE.findall(text)
        if not m:
            return {"error": "no python code block in response",
                    "usage": _usage(r), **tx}
        code = m[-1].strip()

        conf, notes = -1.0, ""
        cm = CONF_RE.search(text)
        if cm:
            try:
                d = json.loads(cm.group(1))
                conf = float(d.get("confidence", -1))
                notes = str(d.get("notes", ""))[:300]
            except (json.JSONDecodeError, ValueError):
                pass

        return {"code": code, "llm_confidence": conf, "notes": notes,
                "logprob_conf": _deepconf(choice), "usage": _usage(r), **tx}

    async def assess_node(self, prompt: str) -> dict:
        """NodeAssessment via DeepSeek JSON mode (no schema enforcement, so the
        scorer system prompt enumerates the JSON keys; we validate with pydantic)."""
        try:
            r = await self._chat(
                [{"role": "system", "content": _SCORER_SYSTEM},
                 {"role": "user", "content": prompt}],
                thinking={"type": "disabled"},
                response_format={"type": "json_object"})
        except Exception as e:
            return {"error": f"api: {e}", "system": _SCORER_SYSTEM, "prompt": prompt}
        text = r.choices[0].message.content or ""
        tx = {"system": _SCORER_SYSTEM, "prompt": prompt,
              "raw_response": text, "reasoning": _reasoning(r.choices[0])}
        try:
            parsed = NodeAssessment.model_validate_json(text)
        except Exception:
            mm = JSON_OBJ_RE.search(text)
            if not mm:
                return {"error": "no JSON object in scorer reply",
                        "usage": _usage(r), **tx}
            try:
                parsed = NodeAssessment.model_validate(json.loads(mm.group(0)))
            except Exception as e:
                return {"error": f"parse: {e}", "usage": _usage(r), **tx}
        return {"assessment": parsed, "usage": _usage(r), **tx}


# scorer system prompt is injected by search/runner via set_scorer_system()
_SCORER_SYSTEM = ""


def set_scorer_system(text: str):
    global _SCORER_SYSTEM
    _SCORER_SYSTEM = text


def _reasoning(choice) -> str:
    """Best-effort extraction of the model's thinking/reasoning content."""
    msg = getattr(choice, "message", None)
    if msg is None:
        return ""
    rc = getattr(msg, "reasoning_content", None)
    if rc is None:
        rc = (getattr(msg, "model_extra", {}) or {}).get("reasoning_content", "")
    return rc or ""


def _usage(r) -> dict:
    u = getattr(r, "usage", None)
    if not u:
        return {}
    extra = getattr(u, "model_extra", {}) or {}
    cache_hit = getattr(u, "prompt_cache_hit_tokens", None)
    if cache_hit is None:
        cache_hit = extra.get("prompt_cache_hit_tokens", 0)
    details = getattr(u, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", 0) if details else 0
    return {
        "input_tokens": getattr(u, "prompt_tokens", 0) or 0,
        "output_tokens": getattr(u, "completion_tokens", 0) or 0,
        "cached_tokens": cache_hit or 0,
        "thought_tokens": reasoning or 0,
    }


def _deepconf(choice, window: int = 128) -> float | None:
    """Offline DeepConf-style confidence: minimum sliding-window mean logprob
    over the answer tokens (higher = more confident)."""
    lp = getattr(choice, "logprobs", None)
    content = getattr(lp, "content", None) if lp else None
    if not content:
        return None
    lps = [t.logprob for t in content if t.logprob is not None]
    if not lps:
        return None
    if len(lps) <= window:
        return float(sum(lps) / len(lps))
    best = math.inf
    s = sum(lps[:window])
    best = min(best, s / window)
    for i in range(window, len(lps)):
        s += lps[i] - lps[i - window]
        best = min(best, s / window)
    return float(best)


# ---------------------------------------------------------------- probe CLI

async def _probe(model: str):
    from harness.prompts import system_prompt, SCORER_SYSTEM
    set_scorer_system(SCORER_SYSTEM)
    llm = LLMClient(model=model, max_concurrent=4)
    meta = await llm.setup(system_prompt("probe hardware"))
    print("capabilities:", json.dumps(meta, indent=2))

    # JSON-mode scorer check
    res = await llm.assess_node(
        "OPERATOR: vector_add (category: elementwise, memory-bound)\n"
        "HARDWARE PEAK BANDWIDTH: 1790 GB/s\nKERNEL:\n```python\npass\n```\n"
        "MEASUREMENT: correct=True speedup=1.0x achieved_bw=1700GB/s (95% of peak)")
    if "assessment" in res:
        print("json scorer: OK ->", res["assessment"].model_dump())
    else:
        print("json scorer: FAILED ->", res.get("error"))

    # generation format + cache-hit check (system prefix was sent in setup)
    res = await llm.generate_candidate(
        "OPERATOR: vector_add\nSIGNATURE: triton_run(x: f32[16M], y: f32[16M]) "
        "-> f32[16M]\nTOLERANCE: rtol=1e-05, atol=1e-05\n"
        "ASSIGNED STRATEGY: seed — write your best initial Triton implementation.\n"
        "Produce the module now, following the OUTPUT FORMAT exactly.")
    if "code" in res:
        u = res["usage"]
        print(f"generation: OK  ({len(res['code'])} chars, "
              f"verbalized_conf={res['llm_confidence']}, "
              f"logprob_conf={res['logprob_conf']}, usage={u})")
        if u.get("cached_tokens", 0) > 0:
            print(f"context cache: HIT ({u['cached_tokens']} tokens served from cache)")
        else:
            print("context cache: cold (expected on first prefix use; rerun to see hits)")
    else:
        print("generation: FAILED ->", res.get("error"))
    await llm.close()


if __name__ == "__main__":
    if "--probe" in sys.argv:
        model = sys.argv[sys.argv.index("--model") + 1] \
            if "--model" in sys.argv else "deepseek-v4-flash"
        asyncio.run(_probe(model))
