"""Unified LLM client for AutoDrive Gym.

Modelled after the kube-sre-gym winner approach:
three clean backends selected by a single env var, no silent mock fallback.

Config (set ONE of these):
  LLM_BACKEND=hf         → HuggingFace Inference API  (use your HF credits)
  LLM_BACKEND=groq       → Groq API                   (fast, free tier)
  LLM_BACKEND=openai     → OpenAI-compatible endpoint  (vLLM / OpenAI)

HF backend env vars:
  HF_TOKEN               — your HuggingFace token (hf_xxx)
  LLM_MODEL              — model ID  (default: Qwen/Qwen2.5-72B-Instruct)

Groq backend env vars:
  GROQ_API_KEY           — your Groq API key
  LLM_MODEL              — model name (default: llama-3.1-8b-instant)

OpenAI backend env vars:
  OPENAI_API_KEY         — API key  (or any string for local vLLM)
  LLM_BASE_URL           — base URL (default: https://api.openai.com/v1)
  LLM_MODEL              — model name (default: gpt-4o-mini)

Auto-detection order (when LLM_BACKEND is not set):
  1. GROQ_API_KEY present  → groq
  2. HF_TOKEN present      → hf
  3. OPENAI_API_KEY present → openai
  4. none                  → mock (limited built-in responses, training still runs)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_json(text: str) -> Dict[str, Any]:
    """Extract JSON from LLM response, handling markdown code fences."""
    text = text.strip()
    # Try direct parse first
    try:
        return json.loads(text)
    except Exception:
        pass
    # Strip ```json ... ``` wrappers
    fence = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except Exception:
            pass
    # Last resort: find first {...} block
    try:
        start = text.index("{")
        end   = text.rindex("}")
        return json.loads(text[start : end + 1])
    except Exception:
        return {"text": text}


def _mock_response(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """Minimal deterministic fallback when no LLM backend is available."""
    combined = (system_prompt + user_prompt).lower()
    if "score" in combined or "grade" in combined:
        return {"score": 0.5, "feedback": "no llm backend configured"}
    if "action" in combined or "decide" in combined:
        return {"action": "wait", "value": 0.0, "reasoning": "no llm backend"}
    if "intent" in combined:
        return {"dominant_scene_intent": "unknown", "agents": []}
    if "negotiation" in combined or "negotiate" in combined:
        return {"outcome": "defer", "priority": "unknown"}
    return {"score": 0.0, "feedback": "no llm backend configured"}


# ── main client ───────────────────────────────────────────────────────────────

class LLMClient:
    """Provider-aware JSON-oriented LLM client.

    Usage::

        llm = LLMClient()
        result = llm.chat_json(system_prompt, user_prompt, temperature=0.2)
    """

    def __init__(self):
        # ── backend selection ──────────────────────────────────────────────
        explicit = os.environ.get("LLM_BACKEND", "").lower()
        if explicit:
            self.backend = explicit
        elif os.environ.get("GROQ_API_KEY"):
            self.backend = "groq"
        elif os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY"):
            self.backend = "hf"
        elif os.environ.get("OPENAI_API_KEY"):
            self.backend = "openai"
        else:
            self.backend = "mock"

        # ── model selection ────────────────────────────────────────────────
        _defaults = {
            "hf":     "Qwen/Qwen2.5-72B-Instruct",
            "groq":   "llama-3.1-8b-instant",
            "openai": "gpt-4o-mini",
            "mock":   "mock",
        }
        self.model = (
            os.environ.get("LLM_MODEL")
            or os.environ.get("HF_MODEL")     # legacy
            or os.environ.get("MODEL_ID")      # legacy
            or os.environ.get("GROQ_MODEL")   # legacy
            or _defaults.get(self.backend, "mock")
        )

        self._client = None

        if self.backend == "hf":
            from huggingface_hub import InferenceClient
            token = (
                os.environ.get("HF_TOKEN")
                or os.environ.get("HUGGINGFACE_API_KEY")
                or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
            )
            # kube-style: use `token=` not `api_key=` (older hub versions need this)
            self._client = InferenceClient(token=token)
            logger.info("LLM backend: HuggingFace Inference API  model=%s", self.model)

        elif self.backend == "groq":
            from groq import Groq
            self._client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            logger.info("LLM backend: Groq  model=%s", self.model)

        elif self.backend == "openai":
            from openai import OpenAI
            self._client = OpenAI(
                base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
                api_key=os.environ.get("OPENAI_API_KEY", "local"),
            )
            logger.info("LLM backend: OpenAI-compatible  model=%s  base=%s",
                        self.model, os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"))

        else:
            logger.warning(
                "LLM backend: mock  (set LLM_BACKEND + credentials to enable real LLM calls)\n"
                "  For HF credits:  set LLM_BACKEND=hf   HF_TOKEN=hf_xxx\n"
                "  For Groq (free): set LLM_BACKEND=groq GROQ_API_KEY=gsk_xxx\n"
                "  Then re-run with --mode pipeline"
            )

    # ── public API ────────────────────────────────────────────────────────────

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 256,
    ) -> Dict[str, Any]:
        """Call LLM and return parsed JSON dict.  Falls back to mock on error."""
        if self.backend == "mock" or self._client is None:
            return _mock_response(system_prompt, user_prompt)
        try:
            text = self._chat(system_prompt, user_prompt, temperature, max_tokens)
            return _parse_json(text)
        except Exception as exc:
            logger.warning("LLM call failed (%s). Using mock response. Error: %s",
                           self.backend, exc)
            return _mock_response(system_prompt, user_prompt)

    # ── internal dispatch ─────────────────────────────────────────────────────

    def _chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        if self.backend == "hf":
            return self._chat_hf(system_prompt, user_prompt, temperature, max_tokens)
        if self.backend == "groq":
            return self._chat_groq(system_prompt, user_prompt, temperature, max_tokens)
        return self._chat_openai(system_prompt, user_prompt, temperature, max_tokens)

    def _chat_hf(self, system: str, user: str, temperature: float, max_tokens: int) -> str:
        # HF InferenceClient.chat_completion — kube-style, with retry
        for attempt in range(3):
            try:
                resp = self._client.chat_completion(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:
                err = str(exc).lower()
                # Non-retriable errors
                if any(x in err for x in ("401", "403", "404", "model not found",
                                           "quota", "billing", "unauthorized")):
                    raise
                if attempt < 2:
                    wait = 2 ** attempt
                    logger.warning("HF transient error (attempt %d/3), retrying in %ds: %s",
                                   attempt + 1, wait, exc)
                    time.sleep(wait)
                else:
                    raise

    def _chat_groq(self, system: str, user: str, temperature: float, max_tokens: int) -> str:
        for attempt in range(3):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:
                err = str(exc).lower()
                if any(x in err for x in ("401", "403", "invalid_api_key", "quota")):
                    raise
                if attempt < 2:
                    wait = 2 ** attempt
                    logger.warning("Groq transient error (attempt %d/3), retrying in %ds: %s",
                                   attempt + 1, wait, exc)
                    time.sleep(wait)
                else:
                    raise

    def _chat_openai(self, system: str, user: str, temperature: float, max_tokens: int) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""
