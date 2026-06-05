"""Abstract environment contract."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from skillrl.llm.base import BaseLLMClient
from skillrl.types import RolloutResult


class SkillEnv(ABC):
    """User-implemented contract for a skill-trainable task.

    The trainer asks the env for items by *split name* and rolls them
    out one by one with a target LLM client.  Items are plain dicts —
    schema is up to the env, the only convention is that ``id`` (when
    present) is used for logging and de-duplication.
    """

    #: Optional list of task-type tags reported in the test summary.
    task_types: list[str] = []

    #: Optional environment name used in logs.
    name: str = "skill_env"

    @abstractmethod
    def get_items(self, split: str) -> list[dict]:
        """Return all dataset items for ``split`` (one of train/val/test)."""

    @abstractmethod
    def rollout_one(
        self,
        *,
        item: dict,
        skill: str,
        target_client: BaseLLMClient,
    ) -> RolloutResult:
        """Run a single task with the given skill, return a scored result.

        ``hard`` MUST be 0/1 (exact-match correctness).  ``soft`` MUST be
        in ``[0, 1]`` (continuous credit).  When the env has no notion of
        soft scoring, set ``soft = float(hard)``.
        """

    # ── Hooks (optional overrides) ───────────────────────────────────

    def get_initial_skill(self) -> str:
        """Return the seed skill document.  Default: empty string."""
        return ""

    def get_task_types(self) -> list[str]:
        return list(self.task_types)
