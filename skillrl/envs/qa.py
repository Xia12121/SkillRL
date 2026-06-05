"""A minimal single-turn QA reference environment.

Each item has the shape::

    {
        "id": "...",
        "question": "...",
        "answers": ["expected", "alt phrasing", ...],   # any one match counts
        "context": "(optional) extra info shown to the agent",
        "task_type": "(optional) tag for reporting"
    }

The target LLM is asked to answer the question conditioned on the
current skill document (used as the system prompt).  Scoring is
case-insensitive substring match: if any expected answer appears as a
substring of the prediction, ``hard = 1``.
"""
from __future__ import annotations

from typing import Iterable

from skillrl.envs.base import SkillEnv
from skillrl.llm.base import BaseLLMClient
from skillrl.types import RolloutResult


def _normalise(s: str) -> str:
    return " ".join((s or "").lower().split())


def _any_match(prediction: str, candidates: Iterable[str]) -> bool:
    pred = _normalise(prediction)
    if not pred:
        return False
    for c in candidates:
        cn = _normalise(c)
        if cn and (cn == pred or cn in pred):
            return True
    return False


def _soft_overlap(prediction: str, candidates: Iterable[str]) -> float:
    """Token-level F1 against the best matching candidate."""
    pred_toks = _normalise(prediction).split()
    if not pred_toks:
        return 0.0
    best = 0.0
    for c in candidates:
        gold_toks = _normalise(c).split()
        if not gold_toks:
            continue
        common = set(pred_toks) & set(gold_toks)
        if not common:
            continue
        precision = len(common) / len(pred_toks)
        recall = len(common) / len(gold_toks)
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best:
            best = f1
    return best


class SimpleQAEnv(SkillEnv):
    """A reference QA env that needs no external dependencies."""

    name = "simple_qa"

    def __init__(
        self,
        *,
        train_items: list[dict],
        val_items: list[dict],
        test_items: list[dict],
        max_completion_tokens: int = 512,
        initial_skill: str = "",
    ) -> None:
        self._splits = {
            "train": list(train_items),
            "val": list(val_items),
            "test": list(test_items),
        }
        self.max_completion_tokens = max_completion_tokens
        self._initial_skill = initial_skill
        # Auto-derive task types if items provide them
        types: set[str] = set()
        for split in self._splits.values():
            for it in split:
                tt = it.get("task_type")
                if tt:
                    types.add(str(tt))
        self.task_types = sorted(types)

    # ── SkillEnv interface ───────────────────────────────────────────

    def get_initial_skill(self) -> str:
        return self._initial_skill

    def get_items(self, split: str) -> list[dict]:
        if split not in self._splits:
            raise KeyError(
                f"Unknown split {split!r}; available: {list(self._splits)}"
            )
        return self._splits[split]

    def rollout_one(
        self,
        *,
        item: dict,
        skill: str,
        target_client: BaseLLMClient,
    ) -> RolloutResult:
        question = str(item.get("question", "") or "")
        context = str(item.get("context", "") or "")
        answers = item.get("answers") or []
        if isinstance(answers, str):
            answers = [answers]

        system = skill or "You are a helpful assistant. Answer the question concisely."
        user_parts = []
        if context:
            user_parts.append(f"Context:\n{context}")
        user_parts.append(f"Question: {question}")
        user_parts.append("Answer:")
        user = "\n\n".join(user_parts)

        text, _usage = target_client.chat(
            system=system,
            user=user,
            max_completion_tokens=self.max_completion_tokens,
            stage="rollout",
            retries=2,
        )

        prediction = (text or "").strip()
        hard = 1 if _any_match(prediction, answers) else 0
        soft = float(hard)
        if not hard and answers:
            soft = round(_soft_overlap(prediction, answers), 4)

        fail_reason = "" if hard else (
            f"prediction does not match any reference answer "
            f"(pred={prediction[:80]!r}, refs={[str(a)[:60] for a in answers[:3]]})"
        )

        return RolloutResult(
            id=str(item.get("id", "")),
            hard=hard,
            soft=soft,
            n_turns=1,
            fail_reason=fail_reason,
            task_type=str(item.get("task_type", "") or ""),
            task_description=str(item.get("task_description", "") or ""),
            predicted_answer=prediction,
            question=question,
            reference_text=" | ".join(str(a) for a in answers),
            target_system_prompt=system,
            target_user_prompt=user,
            conversation=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "assistant", "content": prediction},
            ],
        )
