"""Stage ④ Select — LLM-driven gradient clipping for textual edits.

Given a single merged :class:`~skillrl.types.Patch` whose edit count
exceeds the current edit budget ``L``, ask an optimizer LLM to rank the
edits by importance and keep only the top ``L``.  The fast path (when
``len(edits) <= L``) skips the LLM entirely.
"""
from __future__ import annotations

from skillrl.core.utils import extract_json
from skillrl.llm.base import BaseLLMClient
from skillrl.prompts import load_prompt
from skillrl.types import Edit, Patch


def _describe_edit(idx: int, e: Edit, max_chars: int = 500) -> str:
    parts = [f"op={e.op}"]
    if e.target:
        parts.append(f"target={e.target!r}")
    if e.content:
        parts.append(f"content={e.content!r}")
    if e.support_count is not None:
        parts.append(f"support={e.support_count}")
    if e.source_type:
        parts.append(f"source={e.source_type}")
    text = "  ".join(parts)
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return f"[{idx}] {text}"


def rank_and_select(
    *,
    optimizer_client: BaseLLMClient,
    skill_content: str,
    patch: Patch,
    max_edits: int,
    max_completion_tokens: int = 2048,
    prompt_overrides: dict[str, str] | None = None,
    verbose: bool = True,
) -> Patch:
    """Keep the top-``max_edits`` edits, ranked by an optimizer LLM.

    If the patch already fits the budget, returns it unchanged.

    On any LLM/parse failure, falls back to a deterministic truncation
    that prefers high ``support_count`` and failure-driven edits.
    """
    edits = list(patch.edits)
    if len(edits) <= max_edits:
        return patch

    edits_desc = "\n".join(_describe_edit(i, e) for i, e in enumerate(edits))
    user = (
        f"## Current Skill\n{skill_content}\n\n"
        f"## Edit Pool ({len(edits)} edits, budget={max_edits})\n{edits_desc}\n\n"
        f"Select the {max_edits} most important edits. "
        f"Return their 0-based indices in priority order."
    )
    text, _usage = optimizer_client.chat(
        system=load_prompt("ranking", overrides=prompt_overrides),
        user=user,
        max_completion_tokens=max_completion_tokens,
        stage="ranking",
        retries=3,
    )
    obj = extract_json(text)

    if obj and "selected_indices" in obj:
        seen: set[int] = set()
        picked: list[Edit] = []
        for idx in obj.get("selected_indices") or []:
            if isinstance(idx, int) and 0 <= idx < len(edits) and idx not in seen:
                picked.append(edits[idx])
                seen.add(idx)
            if len(picked) >= max_edits:
                break
        if picked:
            if verbose:
                print(
                    f"    [select] LLM-ranked: {len(edits)} → {len(picked)} "
                    f"(budget={max_edits})"
                )
            return Patch(
                edits=picked,
                reasoning=(patch.reasoning or "")
                + f" [optimizer-ranked: kept {len(picked)}/{len(edits)}]",
                ranking_details=obj,
            )

    # Fallback: deterministic priority — failure first, then by support_count desc
    def _key(e: Edit) -> tuple:
        is_failure = 1 if e.source_type == "failure" else 0
        support = e.support_count or 0
        return (-is_failure, -support)

    sorted_edits = sorted(edits, key=_key)
    if verbose:
        print(
            f"    [select] fallback truncation: {len(edits)} → {max_edits} "
            f"(prioritising failure / high-support edits)"
        )
    return Patch(
        edits=sorted_edits[:max_edits],
        reasoning=(patch.reasoning or "")
        + f" [fallback truncated {len(edits)}->{max_edits}]",
    )


__all__ = ["rank_and_select"]
