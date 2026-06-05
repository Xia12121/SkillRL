"""Validation gating — Stage ⑥ of the skillrl pipeline.

A candidate skill is *accepted* only when its score on the held-out
selection split *strictly improves* upon the running ``current_score``.
The metric is configurable:

* ``hard``   : exact-match accuracy (paper default).
* ``soft``   : continuous per-item reward.
* ``mixed``  : ``(1 - w) * hard + w * soft``.

The gate also tracks the global ``best_skill`` separately, so the
trainer can deploy the all-time-best skill at the end of training.
"""
from __future__ import annotations

from typing import Literal

from skillrl.types import GateResult


GateMetric = Literal["hard", "soft", "mixed"]


def select_gate_score(
    hard: float,
    soft: float,
    metric: GateMetric = "hard",
    mixed_weight: float = 0.5,
) -> float:
    """Combine ``(hard, soft)`` into a single gate score per ``metric``."""
    if metric == "hard":
        return float(hard)
    if metric == "soft":
        return float(soft)
    if metric == "mixed":
        w = float(mixed_weight)
        if not 0.0 <= w <= 1.0:
            raise ValueError(f"mixed_weight must be in [0, 1], got {w}")
        return (1.0 - w) * float(hard) + w * float(soft)
    raise ValueError(f"Unknown gate metric: {metric!r}")


def evaluate_gate(
    *,
    candidate_skill: str,
    cand_hard: float,
    current_skill: str,
    current_score: float,
    best_skill: str,
    best_score: float,
    best_step: int,
    global_step: int,
    cand_soft: float = 0.0,
    metric: GateMetric = "hard",
    mixed_weight: float = 0.5,
) -> GateResult:
    """Decide accept / reject and update ``current`` / ``best`` skill state.

    The decision rule is:

    * If ``cand_score > best_score`` → ``accept_new_best``: replace
      both ``current`` and ``best``.
    * Else if ``cand_score > current_score`` → ``accept``: replace
      ``current`` only.
    * Else → ``reject``: keep both unchanged.
    """
    cand_score = select_gate_score(cand_hard, cand_soft, metric, mixed_weight)

    if cand_score > best_score:
        return GateResult(
            action="accept_new_best",
            current_skill=candidate_skill,
            current_score=cand_score,
            best_skill=candidate_skill,
            best_score=cand_score,
            best_step=global_step,
        )
    if cand_score > current_score:
        return GateResult(
            action="accept",
            current_skill=candidate_skill,
            current_score=cand_score,
            best_skill=best_skill,
            best_score=best_score,
            best_step=best_step,
        )
    return GateResult(
        action="reject",
        current_skill=current_skill,
        current_score=current_score,
        best_skill=best_skill,
        best_score=best_score,
        best_step=best_step,
    )
