"""Abstract LLM client contract used by both the optimizer and target roles."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseLLMClient(ABC):
    """Minimal interface every backend must satisfy.

    A client is asked to produce a single completion given a system
    prompt and a user prompt.  Conversation history (multi-turn) is
    handled by the env adapter, not by the client — this keeps the
    abstraction trivially mockable.
    """

    #: Human-readable model identifier (used for logging / artifacts).
    model: str = ""

    @abstractmethod
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
        """Run a single chat completion.

        Parameters
        ----------
        system, user:
            System prompt and user message.
        max_completion_tokens:
            Token budget for the response.
        temperature:
            Sampling temperature; ``None`` means "let the backend decide".
        stage:
            Free-form tag for the calling pipeline stage
            (``"analyst" | "merge" | "ranking" | "rollout"``).
            Implementations may use this for telemetry.
        retries:
            Network-level retry count.

        Returns
        -------
        text, usage:
            ``text`` is the assistant message content (may be empty).
            ``usage`` is a token-accounting dict with at least
            ``prompt_tokens``, ``completion_tokens``, ``total_tokens``.
        """
        raise NotImplementedError
