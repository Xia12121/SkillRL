"""Training configuration for skillrl — a TRL-like dataclass.

Mirrors the paper-default protocol of SkillOpt (Yang et al., 2026) while
exposing only the knobs needed for the 1.0 core algorithm.  Advanced
features such as slow update, meta skill, autonomous LR, accumulation,
codex/claude-code execution backends and full-rewrite update modes are
intentionally omitted from 1.0 — they are future extension points.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal, Any


GateMetric = Literal["hard", "soft", "mixed"]
LRSchedulerMode = Literal["constant", "linear", "cosine"]


@dataclass
class SkillOptConfig:
    """Training arguments for :class:`~skillrl.trainer.SkillOptTrainer`.

    The defaults reproduce the SkillOpt paper protocol: 4 epochs,
    rollout batch 40, reflection minibatch 8, textual learning rate 4
    with cosine decay, strict hard-validation gating.

    Parameters
    ----------
    num_epochs:
        Number of training epochs.
    batch_size:
        Number of training items rolled out per optimization step.
    minibatch_size:
        Number of trajectories grouped per analyst (Reflect) call.
    merge_batch_size:
        Number of patches merged together at each level of the
        hierarchical Aggregate stage.
    edit_budget:
        Initial / maximum textual learning rate — the cap on edits
        applied per optimization step.  Acts as a soft trust region.
    min_edit_budget:
        Lower bound for decay schedules.
    lr_scheduler:
        Edit-budget schedule: ``constant`` | ``linear`` | ``cosine``.
    gate_metric:
        Validation-gate metric: ``hard`` (exact match) | ``soft``
        (continuous reward) | ``mixed``.
    gate_mixed_weight:
        Weight ``w`` of the soft component when ``gate_metric='mixed'``.
        Final score = ``(1 - w) * hard + w * soft``.
    failure_only:
        If True, only failed trajectories drive analyst patches.
    workers:
        Thread workers for parallel rollout / parallel analyst calls.
    seed:
        Random seed.
    out_root:
        Output directory.  History, per-step artifacts, candidate
        skills and ``best_skill.md`` are written here.  Auto-resume
        reads from this directory if it already contains a run.
    selection_split:
        Name of the split used for the validation gate.  Defaults to
        ``"val"`` (the SkillOpt paper uses ``valid_seen``).
    test_split:
        Name of the held-out test split for the final report.
    eval_test:
        Whether to run the final test evaluation at the end of
        training.
    save_every_step:
        If True, dump per-step ``skill_vXXXX.md`` snapshots to
        ``<out_root>/skills/`` for inspection.
    optimizer_max_completion_tokens:
        Token budget for analyst / merge / ranking calls.
    verbose:
        Print detailed per-stage progress.
    extras:
        Free-form extra config — passed through to environments and
        custom prompt templates.
    """

    # ── Training schedule ────────────────────────────────────────────
    num_epochs: int = 4
    batch_size: int = 40
    minibatch_size: int = 8
    merge_batch_size: int = 8

    # ── Optimizer (textual LR) ───────────────────────────────────────
    edit_budget: int = 4
    min_edit_budget: int = 2
    lr_scheduler: LRSchedulerMode = "cosine"

    # ── Validation gate ──────────────────────────────────────────────
    gate_metric: GateMetric = "hard"
    gate_mixed_weight: float = 0.5

    # ── Reflect stage ────────────────────────────────────────────────
    failure_only: bool = False

    # ── Concurrency / runtime ────────────────────────────────────────
    workers: int = 8
    seed: int = 42
    out_root: str = "outputs/skillrl_run"

    # ── Evaluation splits ────────────────────────────────────────────
    selection_split: str = "val"
    test_split: str = "test"
    eval_test: bool = True

    # ── Persistence / observability ──────────────────────────────────
    save_every_step: bool = True
    optimizer_max_completion_tokens: int = 4096
    verbose: bool = True

    # ── Free-form extras (forwarded to env/prompts) ──────────────────
    extras: dict[str, Any] = field(default_factory=dict)

    # ── Helpers ──────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.minibatch_size <= 0:
            raise ValueError(f"minibatch_size must be > 0, got {self.minibatch_size}")
        if self.merge_batch_size < 2:
            raise ValueError(f"merge_batch_size must be >= 2, got {self.merge_batch_size}")
        if self.edit_budget <= 0:
            raise ValueError(f"edit_budget must be > 0, got {self.edit_budget}")
        if self.min_edit_budget <= 0 or self.min_edit_budget > self.edit_budget:
            raise ValueError(
                f"min_edit_budget must be in (0, edit_budget], "
                f"got {self.min_edit_budget} (edit_budget={self.edit_budget})"
            )
        if self.lr_scheduler not in ("constant", "linear", "cosine"):
            raise ValueError(
                f"lr_scheduler must be one of constant/linear/cosine, "
                f"got {self.lr_scheduler!r}"
            )
        if self.gate_metric not in ("hard", "soft", "mixed"):
            raise ValueError(
                f"gate_metric must be one of hard/soft/mixed, got {self.gate_metric!r}"
            )
        if not 0.0 <= self.gate_mixed_weight <= 1.0:
            raise ValueError(
                f"gate_mixed_weight must be in [0, 1], got {self.gate_mixed_weight}"
            )

    def to_dict(self) -> dict:
        return asdict(self)
