"""Standardised I/O types shared across the 6-stage skillrl pipeline.

All dataclasses support round-trip ``from_dict`` / ``to_dict`` conversion,
mirroring the SkillOpt reference implementation so JSON artifacts stay
interoperable.  See the ReflACT paper for the formal stage contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields as dc_fields
from typing import Any, Literal


# ── Atomic edit ──────────────────────────────────────────────────────────

EditOp = Literal["append", "insert_after", "replace", "delete"]


@dataclass
class Edit:
    """A single textual edit on the skill document.

    Used across Reflect → Aggregate → Select → Update.

    Attributes
    ----------
    op:
        Operation kind: ``append`` | ``insert_after`` | ``replace`` | ``delete``.
    content:
        New text to write (ignored for ``delete``).
    target:
        Anchor text to locate.  Required for ``replace`` / ``delete`` /
        ``insert_after``; ignored for ``append``.
    support_count:
        Number of trajectories that supported this edit (set during
        Reflect).  Used as a tie-breaker for ranking.
    source_type:
        ``"failure"`` or ``"success"`` — indicates which analyst
        produced this edit.
    merge_level:
        Level in the hierarchical Aggregate tree at which the edit was
        produced (1 = first merge level).
    """

    op: EditOp
    content: str = ""
    target: str = ""
    support_count: int | None = None
    source_type: Literal["failure", "success"] | None = None
    merge_level: int | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "Edit":
        return cls(
            op=d.get("op", "append"),
            content=d.get("content", "") or "",
            target=d.get("target", "") or "",
            support_count=d.get("support_count"),
            source_type=d.get("source_type"),
            merge_level=d.get("merge_level"),
        )

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"op": self.op, "content": self.content}
        if self.target:
            out["target"] = self.target
        if self.support_count is not None:
            out["support_count"] = self.support_count
        if self.source_type is not None:
            out["source_type"] = self.source_type
        if self.merge_level is not None:
            out["merge_level"] = self.merge_level
        return out


# ── Patch (a bag of edits) ───────────────────────────────────────────────


@dataclass
class Patch:
    """A set of edits with reasoning.

    Output of Aggregate (③) and Select (④); input to Update (⑤).
    """

    edits: list[Edit] = field(default_factory=list)
    reasoning: str = ""
    ranking_details: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "Patch":
        edits_raw = d.get("edits", []) or []
        edits = [
            Edit.from_dict(e) if isinstance(e, dict) else e for e in edits_raw
        ]
        return cls(
            edits=edits,
            reasoning=d.get("reasoning", "") or "",
            ranking_details=d.get("ranking_details"),
        )

    def to_dict(self) -> dict:
        out: dict[str, Any] = {
            "reasoning": self.reasoning,
            "edits": [e.to_dict() if isinstance(e, Edit) else e for e in self.edits],
        }
        if self.ranking_details is not None:
            out["ranking_details"] = self.ranking_details
        return out


# ── Reflect output (with provenance) ─────────────────────────────────────


@dataclass
class RawPatch:
    """Per-minibatch analyst output, wrapping a :class:`Patch` with metadata."""

    patch: Patch
    source_type: Literal["failure", "success"] = "failure"
    batch_size: int = 0

    @classmethod
    def from_dict(cls, d: dict | None) -> "RawPatch | None":
        if d is None:
            return None
        inner = d.get("patch", d)
        if not isinstance(inner, dict):
            return None
        return cls(
            patch=Patch.from_dict(inner),
            source_type=d.get("source_type", "failure"),
            batch_size=int(d.get("batch_size", 0)),
        )

    def to_dict(self) -> dict:
        return {
            "patch": self.patch.to_dict(),
            "source_type": self.source_type,
            "batch_size": self.batch_size,
        }


# ── Rollout result ───────────────────────────────────────────────────────


@dataclass
class RolloutResult:
    """Result of a single task rollout (Stage ①).

    The ``hard`` field is binary (0/1) — exact-match correctness.  The
    ``soft`` field is a continuous reward in [0, 1] used by soft/mixed
    gating.  Environment-specific information lives in ``extras``.
    """

    id: str
    hard: int = 0
    soft: float = 0.0
    n_turns: int = 0
    fail_reason: str = ""
    task_type: str = ""
    task_description: str = ""
    predicted_answer: str = ""
    question: str = ""
    reference_text: str = ""
    target_system_prompt: str = ""
    target_user_prompt: str = ""
    conversation: list[dict] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    _KNOWN: frozenset[str] | None = field(
        default=None, init=False, repr=False, compare=False,
    )

    @classmethod
    def _known_fields(cls) -> frozenset[str]:
        if cls._KNOWN is None:
            cls._KNOWN = frozenset(
                f.name for f in dc_fields(cls) if f.name != "_KNOWN"
            )
        return cls._KNOWN

    @classmethod
    def from_dict(cls, d: dict) -> "RolloutResult":
        known = cls._known_fields()
        extras = {k: v for k, v in d.items() if k not in known}
        return cls(
            id=str(d.get("id", "")),
            hard=int(d.get("hard", 0)),
            soft=float(d.get("soft", 0.0)),
            n_turns=int(d.get("n_turns", 0)),
            fail_reason=str(d.get("fail_reason", "") or ""),
            task_type=str(d.get("task_type", "") or ""),
            task_description=str(d.get("task_description", "") or ""),
            predicted_answer=str(d.get("predicted_answer", "") or ""),
            question=str(d.get("question", "") or ""),
            reference_text=str(d.get("reference_text", "") or ""),
            target_system_prompt=str(d.get("target_system_prompt", "") or ""),
            target_user_prompt=str(d.get("target_user_prompt", "") or ""),
            conversation=list(d.get("conversation", []) or []),
            extras=extras,
        )

    def to_dict(self) -> dict:
        out: dict[str, Any] = {
            "id": self.id,
            "hard": self.hard,
            "soft": self.soft,
        }
        for attr in (
            "n_turns", "fail_reason", "task_type", "task_description",
            "predicted_answer", "question", "reference_text",
            "target_system_prompt", "target_user_prompt",
        ):
            val = getattr(self, attr)
            if val:
                out[attr] = val
        if self.conversation:
            out["conversation"] = self.conversation
        out.update(self.extras)
        return out


# ── Validation gate ──────────────────────────────────────────────────────

GateAction = Literal["accept_new_best", "accept", "reject"]


@dataclass
class GateResult:
    """Outcome of the validation gate (Stage ⑥).

    Carries the post-gate ``current`` and ``best`` state so the trainer
    can simply unpack and continue.
    """

    action: GateAction
    current_skill: str
    current_score: float
    best_skill: str
    best_score: float
    best_step: int
