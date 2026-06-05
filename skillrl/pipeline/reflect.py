"""Stage ② Reflect — minibatch trajectory analysis.

Mirrors the SkillOpt design: rollouts are bucketed by outcome (failure /
success), each bucket is sliced into deterministic minibatches of size
``M``, and one analyst LLM call processes each minibatch to emit a
candidate :class:`~skillrl.types.Patch`.

Parallelism is achieved with a thread pool.  Failure analysts and
success analysts run independently.
"""
from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from skillrl.core.utils import extract_json
from skillrl.llm.base import BaseLLMClient
from skillrl.prompts import load_prompt
from skillrl.types import Edit, Patch, RawPatch, RolloutResult


# ── Trajectory formatting ─────────────────────────────────────────────────

_MAX_TRAJ_CHARS = 12_000


def _clip(text: str, limit: int) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    half = limit // 2 - 12
    return text[:half] + "\n... [truncated] ...\n" + text[-half:]


def _fmt_one_trajectory(r: RolloutResult, limit: int = _MAX_TRAJ_CHARS) -> str:
    """Render a single rollout result as a textual block for the analyst."""
    lines: list[str] = []
    lines.append(f"### Task id={r.id} hard={r.hard} soft={r.soft:.2f}")
    if r.task_type:
        lines.append(f"Task type: {r.task_type}")
    if r.task_description:
        lines.append(f"Task: {_clip(r.task_description, 1000)}")
    if r.question:
        lines.append(f"Question: {_clip(r.question, 1000)}")
    if r.reference_text:
        lines.append(f"Reference: {_clip(r.reference_text, 2000)}")
    if r.predicted_answer:
        lines.append(f"Model answer: {_clip(r.predicted_answer, 2000)}")
    if r.fail_reason:
        lines.append(f"Failure reason: {_clip(r.fail_reason, 600)}")
    if r.conversation:
        # Conversation is a list of {role, content} dicts
        try:
            convo = "\n".join(
                f"[{m.get('role', '?')}] {_clip(str(m.get('content', '')), 800)}"
                for m in r.conversation
            )
        except Exception:  # noqa: BLE001
            convo = json.dumps(r.conversation, ensure_ascii=False)[:2000]
        lines.append(f"Conversation:\n{convo}")
    block = "\n".join(lines)
    return _clip(block, limit)


def _fmt_minibatch(items: list[RolloutResult]) -> str:
    return "\n\n---\n\n".join(_fmt_one_trajectory(r) for r in items)


# ── Analyst calls ─────────────────────────────────────────────────────────


def _build_analyst_user_prompt(
    *,
    current_skill: str,
    minibatch: list[RolloutResult],
    edit_budget: int,
    label: str,
    extra_context: str = "",
) -> str:
    parts = [f"## Current Skill\n{current_skill or '(empty — no skill yet)'}"]
    parts.append(
        f"## Edit Budget\nProduce AT MOST {edit_budget} edits "
        "(fewer is fine; do NOT pad)."
    )
    if extra_context:
        parts.append(f"## Additional Context\n{extra_context}")
    parts.append(
        f"## {label} Trajectories ({len(minibatch)} total)\n{_fmt_minibatch(minibatch)}"
    )
    return "\n\n".join(parts)


def _parse_patch_response(text: str) -> Patch | None:
    obj = extract_json(text)
    if not obj:
        return None
    if "edits" not in obj or not isinstance(obj.get("edits", []), list):
        return None
    return Patch.from_dict(obj)


def _run_analyst(
    *,
    optimizer_client: BaseLLMClient,
    system_prompt: str,
    user_prompt: str,
    max_completion_tokens: int,
    stage: str,
    source_type: str,
    minibatch_size: int,
    edit_budget: int,
) -> RawPatch | None:
    text, _usage = optimizer_client.chat(
        system=system_prompt,
        user=user_prompt,
        max_completion_tokens=max_completion_tokens,
        stage=stage,
        retries=3,
    )
    patch = _parse_patch_response(text)
    if patch is None:
        return None

    # Truncate to budget; tag provenance.
    patch.edits = patch.edits[: max(edit_budget, 1)]
    for e in patch.edits:
        if e.support_count is None:
            e.support_count = minibatch_size
        if e.source_type is None:
            e.source_type = source_type  # type: ignore[assignment]
    return RawPatch(patch=patch, source_type=source_type, batch_size=minibatch_size)  # type: ignore[arg-type]


# ── Public API ────────────────────────────────────────────────────────────


def run_minibatch_reflect(
    *,
    rollout_results: list[RolloutResult],
    current_skill: str,
    optimizer_client: BaseLLMClient,
    minibatch_size: int = 8,
    edit_budget: int = 4,
    failure_only: bool = False,
    workers: int = 8,
    seed: int = 42,
    max_completion_tokens: int = 4096,
    prompt_overrides: dict[str, str] | None = None,
    extra_context: str = "",
    verbose: bool = True,
) -> list[RawPatch]:
    """Run the Reflect stage on a batch of rollouts.

    Returns
    -------
    list[RawPatch]
        One :class:`RawPatch` per minibatch that produced a parseable
        analyst response.  Order mixes failure and success outputs.
    """
    failures = [r for r in rollout_results if not r.hard]
    successes = [] if failure_only else [r for r in rollout_results if r.hard]

    rng_f = random.Random(seed)
    rng_s = random.Random(seed + 1)
    rng_f.shuffle(failures)
    rng_s.shuffle(successes)

    def _split(items: list[RolloutResult]) -> list[list[RolloutResult]]:
        return [
            items[i : i + minibatch_size]
            for i in range(0, len(items), minibatch_size)
        ]

    fail_batches = _split(failures)
    succ_batches = _split(successes)

    if verbose:
        print(
            f"    [reflect] failures={len(failures)} ({len(fail_batches)} mb) "
            f"successes={len(successes)} ({len(succ_batches)} mb) "
            f"M={minibatch_size}"
        )

    err_sys = load_prompt("analyst_error", overrides=prompt_overrides)
    suc_sys = load_prompt("analyst_success", overrides=prompt_overrides)

    jobs: list[tuple[str, str, str, list[RolloutResult]]] = []
    for mb in fail_batches:
        if mb:
            user = _build_analyst_user_prompt(
                current_skill=current_skill,
                minibatch=mb,
                edit_budget=edit_budget,
                label="Failed",
                extra_context=extra_context,
            )
            jobs.append(("failure", err_sys, user, mb))
    for mb in succ_batches:
        if mb:
            user = _build_analyst_user_prompt(
                current_skill=current_skill,
                minibatch=mb,
                edit_budget=edit_budget,
                label="Successful",
                extra_context=extra_context,
            )
            jobs.append(("success", suc_sys, user, mb))

    out: list[RawPatch | None] = [None] * len(jobs)

    def _one(idx: int, src: str, sys_p: str, user_p: str, mb: list[RolloutResult]):
        rp = _run_analyst(
            optimizer_client=optimizer_client,
            system_prompt=sys_p,
            user_prompt=user_p,
            max_completion_tokens=max_completion_tokens,
            stage="analyst",
            source_type=src,
            minibatch_size=len(mb),
            edit_budget=edit_budget,
        )
        return idx, rp

    if workers <= 1 or len(jobs) <= 1:
        for i, (src, sys_p, user_p, mb) in enumerate(jobs):
            _, rp = _one(i, src, sys_p, user_p, mb)
            out[i] = rp
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(_one, i, src, sys_p, user_p, mb)
                for i, (src, sys_p, user_p, mb) in enumerate(jobs)
            ]
            for fut in as_completed(futs):
                i, rp = fut.result()
                out[i] = rp

    raw_patches = [rp for rp in out if rp is not None]
    if verbose:
        n_fail = sum(1 for rp in raw_patches if rp.source_type == "failure")
        n_suc = len(raw_patches) - n_fail
        n_edits = sum(len(rp.patch.edits) for rp in raw_patches)
        print(
            f"    [reflect done] {len(raw_patches)} patches "
            f"(failure={n_fail} success={n_suc}, total edits={n_edits})"
        )
    return raw_patches


__all__ = ["run_minibatch_reflect"]
