"""OpenAI / Azure-OpenAI / OpenAI-compatible chat client."""
from __future__ import annotations

import os
import time
from typing import Any

from skillrl.llm.base import BaseLLMClient


class OpenAIChatClient(BaseLLMClient):
    """A thin OpenAI Chat Completions wrapper.

    Works out of the box for:

    * The official OpenAI API (``base_url=None``, ``api_key`` env or arg).
    * Azure OpenAI — pass ``azure_endpoint`` + ``api_version``.
    * Any OpenAI-compatible server (vLLM, Together, MoonshotAI, etc.) —
      pass the appropriate ``base_url`` and ``api_key``.

    Parameters
    ----------
    model:
        Deployment / model name (e.g. ``"gpt-4o-mini"`` or your Azure
        deployment id).
    api_key:
        API key.  Falls back to ``OPENAI_API_KEY`` env var.
    base_url:
        Base URL for the OpenAI-compatible service.  Defaults to the
        official OpenAI endpoint when ``None``.
    azure_endpoint, api_version:
        Set both to use Azure OpenAI; the client will instantiate
        ``AzureOpenAI`` instead of ``OpenAI``.
    default_temperature:
        Default sampling temperature.  ``None`` means "do not pass
        ``temperature`` to the backend" — useful for o-series models
        that reject the parameter.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        default_temperature: float | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.default_temperature = default_temperature
        self.timeout = timeout

        try:
            import openai  # noqa: F401
        except ImportError as exc:  # noqa: BLE001
            raise ImportError(
                "OpenAIChatClient requires the `openai` package. "
                "Install with: pip install openai>=1.40.0"
            ) from exc

        if azure_endpoint:
            from openai import AzureOpenAI

            self._client = AzureOpenAI(
                api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"),
                api_version=api_version or "2024-12-01-preview",
                azure_endpoint=azure_endpoint,
                timeout=timeout,
            )
            self._kind = "azure"
        else:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=api_key or os.getenv("OPENAI_API_KEY"),
                base_url=base_url,
                timeout=timeout,
            )
            self._kind = "openai"

    # ── Public API ─────────────────────────────────────────────────────

    def chat(
        self,
        *,
        system: str,
        user: str,
        max_completion_tokens: int = 4096,
        temperature: float | None = None,
        stage: str = "",
        retries: int = 3,
        **kwargs: Any,
    ) -> tuple[str, dict[str, int]]:
        messages = [
            {"role": "system", "content": system or ""},
            {"role": "user", "content": user or ""},
        ]
        temp = temperature if temperature is not None else self.default_temperature

        last_err: Exception | None = None
        backoff = 1.0
        for attempt in range(max(1, retries)):
            try:
                params: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "max_completion_tokens": max_completion_tokens,
                }
                if temp is not None:
                    params["temperature"] = temp

                resp = self._client.chat.completions.create(**params)

                text = ""
                if resp.choices:
                    text = (resp.choices[0].message.content or "").strip()
                usage = {
                    "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) if resp.usage else 0,
                    "completion_tokens": getattr(resp.usage, "completion_tokens", 0) if resp.usage else 0,
                    "total_tokens": getattr(resp.usage, "total_tokens", 0) if resp.usage else 0,
                    "calls": 1,
                    "stage": stage,
                }
                return text, usage
            except Exception as exc:  # noqa: BLE001
                # Some backends (older models) reject `max_completion_tokens`
                # — retry once with `max_tokens` for compatibility.
                if "max_completion_tokens" in str(exc) and "max_tokens" not in str(exc):
                    try:
                        params2 = dict(params)
                        params2.pop("max_completion_tokens", None)
                        params2["max_tokens"] = max_completion_tokens
                        resp = self._client.chat.completions.create(**params2)
                        text = ""
                        if resp.choices:
                            text = (resp.choices[0].message.content or "").strip()
                        usage = {
                            "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) if resp.usage else 0,
                            "completion_tokens": getattr(resp.usage, "completion_tokens", 0) if resp.usage else 0,
                            "total_tokens": getattr(resp.usage, "total_tokens", 0) if resp.usage else 0,
                            "calls": 1,
                            "stage": stage,
                        }
                        return text, usage
                    except Exception as exc2:  # noqa: BLE001
                        last_err = exc2
                else:
                    last_err = exc
                if attempt + 1 < retries:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 16.0)

        # Exhausted retries — return empty text, do not crash the trainer
        return "", {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
            "stage": stage,
            "error": str(last_err) if last_err else "",
        }
