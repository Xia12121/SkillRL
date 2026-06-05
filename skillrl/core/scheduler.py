"""Edit-budget (textual learning rate) schedulers.

In skillrl the *learning rate* is the maximum number of edits applied
to the skill document at each optimization step.  Exposing the same
PyTorch-style API (``step()``, ``state_dict()`` / ``load_state_dict``)
keeps the abstraction familiar.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod


class LRScheduler(ABC):
    """Base class for edit-budget schedulers."""

    def __init__(self, max_lr: int, min_lr: int, total_steps: int) -> None:
        self.max_lr = int(max_lr)
        self.min_lr = int(min_lr)
        self.total_steps = max(int(total_steps), 1)
        self._current_step = 0

    @abstractmethod
    def _compute_lr(self, step: int) -> int:
        """Return the edit budget for a 1-indexed step."""

    def step(self) -> int:
        self._current_step += 1
        return self._compute_lr(self._current_step)

    def get_lr(self, step: int) -> int:
        return self._compute_lr(step)

    def state_dict(self) -> dict:
        return {"current_step": self._current_step}

    def load_state_dict(self, state: dict) -> None:
        self._current_step = int(state.get("current_step", 0) or 0)


class ConstantScheduler(LRScheduler):
    """Fixed budget throughout training."""

    def _compute_lr(self, step: int) -> int:
        return self.max_lr


class LinearScheduler(LRScheduler):
    """Linear decay from ``max_lr`` to ``min_lr`` over ``total_steps``."""

    def _compute_lr(self, step: int) -> int:
        if self.total_steps <= 1:
            return self.max_lr
        t = min(step, self.total_steps) / self.total_steps
        lr = self.max_lr + (self.min_lr - self.max_lr) * t
        return max(self.min_lr, round(lr))


class CosineScheduler(LRScheduler):
    """Cosine annealing from ``max_lr`` to ``min_lr`` over ``total_steps``."""

    def _compute_lr(self, step: int) -> int:
        if self.total_steps <= 1:
            return self.max_lr
        t = min(step, self.total_steps) / self.total_steps
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * t))
        return max(self.min_lr, round(lr))


_REGISTRY: dict[str, type[LRScheduler]] = {
    "constant": ConstantScheduler,
    "linear": LinearScheduler,
    "cosine": CosineScheduler,
}


def build_scheduler(
    mode: str = "cosine",
    max_lr: int = 4,
    min_lr: int = 2,
    total_steps: int = 8,
) -> LRScheduler:
    """Build an edit-budget scheduler from config parameters."""
    if mode not in _REGISTRY:
        raise ValueError(
            f"Unknown scheduler mode {mode!r}. Available: {list(_REGISTRY)}"
        )
    return _REGISTRY[mode](max_lr=max_lr, min_lr=min_lr, total_steps=total_steps)
