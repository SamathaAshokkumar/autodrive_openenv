"""Unified LLM client for AutoDrive Gym."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict

from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency path
    OpenAI = None

try:
    from groq import Groq
except Exception:  # pragma: no cover - optional dependency path
    Groq = None


class LLMClient:
    """Provider-aware JSON-oriented client.

    Supported providers:
    - `openai`
    - `groq`
    - `hf`
    - fallback mock response when no provider is configured or available
    """

    def __init__(self, provider: str | None = None, api_key: str | None = None):
        backend = os.environ.get("LLM_BACKEND")
        explicit_provider = os.environ.get("LLM_PROVIDER")
        normalized_provider = provider or explicit_provider or backend
        self.provider = (
            normalized_provider
            or ("openai" if os.environ.get("OPENAI_API_KEY") else None)
            or ("groq" if os.environ.get("GROQ_API_KEY") else None)
            or ("hf" if (
                os.environ.get("HF_TOKEN")
                or os.environ.get("HUGGINGFACE_API_KEY")
                or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
            ) else None)
            or "mock"
        ).lower()

        self.api_key = api_key or os.environ.get("LLM_API_KEY")

        self.openai_key = os.environ.get("OPENAI_API_KEY") or self.api_key
        self.openai_model = (
            os.environ.get("OPENAI_MODEL")
            or os.environ.get("LLM_MODEL")
            or "gpt-4o-mini"
        )
        self.openai_base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("LLM_BASE_URL")

        self.groq_key = os.environ.get("GROQ_API_KEY") or self.api_key
        self.groq_model = (
            os.environ.get("GROQ_MODEL")
            or os.environ.get("LLM_MODEL")
            or "llama-3.1-8b-instant"
        )

        self.hf_key = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_API_KEY")
            or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        )
        self.hf_model = (
            os.environ.get("HF_MODEL")
            or os.environ.get("MODEL_ID")
            or os.environ.get("HUGGINGFACE_MODEL")
            or os.environ.get("LLM_MODEL")
            or "Qwen/Qwen3-0.6B"
        )
        self.hf_provider = (
            os.environ.get("HF_PROVIDER")
            or os.environ.get("HUGGINGFACE_PROVIDER")
            or "auto"
        )

        self._disabled_reason = ""
        self._openai_client = None
        self._groq_client = None
        self._hf_client = None
        self._provider_disabled = False

        if self.provider == "openai" and self.openai_key and OpenAI is not None:
            self._openai_client = OpenAI(
                api_key=self.openai_key,
                base_url=self.openai_base_url,
            )
            logger.info("LLM provider: openai (%s)", self.openai_model)
        elif self.provider == "groq" and self.groq_key and Groq is not None:
            self._groq_client = Groq(api_key=self.groq_key)
            logger.info("LLM provider: groq (%s)", self.groq_model)
        elif self.provider == "hf" and self.hf_key:
            self._hf_client = InferenceClient(
                model=self.hf_model,
                provider=self.hf_provider,
                api_key=self.hf_key,
                timeout=30,
            )
            logger.info("LLM provider: hf (%s via %s)", self.hf_model, self.hf_provider)
        else:
            if self.provider == "openai" and OpenAI is None:
                self._disabled_reason = "openai package is not installed"
            elif self.provider == "groq" and Groq is None:
                self._disabled_reason = "groq package is not installed"
            elif self.provider == "hf" and not self.hf_key:
                self._disabled_reason = "HF token is not configured"
            else:
                self._disabled_reason = "no remote provider configured"
            logger.info("LLM provider: mock (%s)", self._disabled_reason)
            self.provider = "mock"

    @staticmethod
    def _parse_json_from_text(text: str) -> Dict[str, Any]:
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            pass

        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if fence_match:
            fenced = fence_match.group(1).strip()
            try:
                return json.loads(fenced)
            except Exception:
                pass

        try:
            start = text.index("{")
            end = text.rindex("}")
            return json.loads(text[start : end + 1])
        except Exception:
            return {"text": text}

    def _chat_openai(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
        completion = self._openai_client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = completion.choices[0].message.content or ""
        return self._parse_json_from_text(text)

    def _chat_groq(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
        completion = self._groq_client.chat.completions.create(
            model=self.groq_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = completion.choices[0].message.content or ""
        return self._parse_json_from_text(text)

    def _chat_hf(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
        try:
            completion = self._hf_client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=self.hf_model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = completion.choices[0].message.content or ""
            return self._parse_json_from_text(text)
        except Exception as chat_exc:
            logger.info(
                "HF chat_completion unavailable for model '%s' via provider '%s'; "
                "trying text_generation fallback. Error: %s",
                self.hf_model,
                self.hf_provider,
                chat_exc,
            )
            text = self._hf_client.text_generation(
                prompt=f"{system_prompt}\n{user_prompt}",
                model=self.hf_model,
                max_new_tokens=max_tokens,
                temperature=temperature,
                return_full_text=False,
            )
            if not isinstance(text, str):
                text = str(text)
            return self._parse_json_from_text(text)

    def _disable_provider(self, reason: str) -> None:
        self._provider_disabled = True
        self._disabled_reason = reason
        logger.warning(reason)

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 256,
    ) -> Dict[str, Any]:
        if self._provider_disabled:
            logger.debug("Mock LLM fallback: %s", self._disabled_reason)
            if "score" in user_prompt.lower():
                return {"score": 0.0, "feedback": "mocked neutral"}
            if "return json" in user_prompt.lower():
                return {"action": "wait", "value": 0.0}
            return {"score": 0.0, "feedback": "mock response"}

        try:
            if self.provider == "openai" and self._openai_client is not None:
                return self._chat_openai(system_prompt, user_prompt, temperature, max_tokens)
            if self.provider == "groq" and self._groq_client is not None:
                return self._chat_groq(system_prompt, user_prompt, temperature, max_tokens)
            if self.provider == "hf" and self._hf_client is not None:
                return self._chat_hf(system_prompt, user_prompt, temperature, max_tokens)
        except Exception as exc:
            error_text = str(exc)
            fatal_markers = [
                "insufficient_quota",
                "exceeded your current quota",
                "incorrect api key",
                "invalid api key",
                "model_not_supported",
                "authentication",
                "401",
                "403",
                "404",
                "410",
            ]
            if any(marker in error_text.lower() for marker in fatal_markers):
                self._disable_provider(
                    f"Provider '{self.provider}' is unavailable for the configured model or account. "
                    f"Falling back to the built-in baseline agent. Error: {error_text}"
                )
            else:
                logger.warning(
                    "Provider '%s' failed for the configured model. Falling back to built-in baseline behavior. Error: %s",
                    self.provider,
                    exc,
                )

        if self._disabled_reason:
            logger.debug("Mock LLM fallback: %s", self._disabled_reason)
        else:
            logger.debug("Mock LLM fallback: no provider response available.")

        if "score" in user_prompt.lower():
            return {"score": 0.0, "feedback": "mocked neutral"}
        if "return json" in user_prompt.lower():
            return {"action": "wait", "value": 0.0}
        return {"score": 0.0, "feedback": "mock response"}


__all__ = ["LLMClient"]
