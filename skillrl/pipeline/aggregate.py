"""Stage ③ Aggregate — hierarchical, failure-first patch merging.

Many ``RawPatch`` objects come out of Reflect.  Aggregate consolidates
them into a single coherent :class:`~skillrl.types.Patch` via repeated
LLM merge calls organised as a balanced tree (configurable
``merge_batch_size``).  Failure-driven and success-driven groups are
merged independently first, then a final merge step combines them with
failure priority.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from skillrl.core.utils import extract_json
from skillrl.llm.base import BaseLLMClient
from skillrl.prompts import load_prompt
from skillrl.types import Edit, Patch, RawPatch


def _patches_to_json(patches: list[Patch]) -> str:
    return json.dumps([p.to_dict() for p in patches], ensure_ascii=False, indent=2)


def _llm_merge_batch(
    *,
    optimizer_client: BaseLLMClient,
    system_prompt: str,
    skill_content: str,
    patches: list[Patch],
    level: int,
    max_completion_tokens: int,
) -> Patch:
    user = (
        f"## Current Skill\n{skill_content}\n\n"
        f"## Patches to merge ({len(patches)} total, merge level {level})\n"
        f"{_patches_to_json(patches)}"
    )
    text, _usage = optimizer_client.chat(
        system=system_prompt,
        user=user,
        max_completion_tokens=max_completion_tokens,
        stage="merge",
        retries=3,
    )
    obj = extract_json(text)
    if obj and isinstance(obj.get("edits"), list):
        merged = Patch.from_dict(obj)
        for e in merged.edits:
            if e.merge_level is None:
                e.merge_level = level
        return merged

    # Fallback: simple concatenation
    all_edits: list[Edit] = []
    for p in patches:
        for e in p.edits:
            if e.merge_level is None:
                e.merge_level = level
            all_edits.append(e)
    return Patch(edits=all_edits, reasoning="fallback concatenation")


def _hierarchical_merge(
    *,
    optimizer_client: BaseLLMClient,
    system_prompt: str,
    skill_content: str,
    patches: list[Patch],
    batch_size: int,
    workers: int,
    max_completion_tokens: int,
    label: str,
    verbose: bool,
) -> Patch:
    if not patches:
        return Patch(edits=[], reasoning="no patches")
    if len(patches) == 1:
        return patches[0]

    current = list(patches)
    level = 0
    while len(current) > 1:
        level += 1
        groups = [current[i : i + batch_size] for i in range(0, len(current), batch_size)]
        if verbose:
            print(
                f"    [aggregate {label}] level={level}  "
                f"{len(current)} → {len(groups)} groups (bs={batch_size})"
            )

        next_level: list[Patch | None] = [None] * len(groups)
        to_merge: list[tuple[int, list[Patch]]] = []
        for idx, g in enumerate(groups):
            if len(g) == 1:
                next_level[idx] = g[0]
            else:
                to_merge.append((idx, g))

        if to_merge:
            with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
                futs = {
                    ex.submit(
                        _llm_merge_batch,
                        optimizer_client=optimizer_client,
                        system_prompt=system_prompt,
                        skill_content=skill_content,
                        patches=g,
                        level=level,
                        max_completion_tokens=max_completion_tokens,
                    ): idx
                    for idx, g in to_merge
                }
                for fut in as_completed(futs):
                    idx = futs[fut]
                    next_level[idx] = fut.result()

        current = [p for p in next_level if p is not None]

    return current[0]


def merge_patches(
    *,
    optimizer_client: BaseLLMClient,
    skill_content: str,
    raw_patches: list[RawPatch],
    batch_size: int = 8,
    workers: int = 8,
    max_completion_tokens: int = 4096,
    prompt_overrides: dict[str, str] | None = None,
    verbose: bool = True,
) -> Patch:
    """Failure-first hierarchical merge of analyst patches.

    Steps
    -----
    1. Independently merge all failure-driven patches.
    2. Independently merge all success-driven patches.
    3. Final merge of the two groups with failure priority.
    """
    failure = [rp.patch for rp in raw_patches if rp.source_type == "failure" and rp.patch.edits]
    success = [rp.patch for rp in raw_patches if rp.source_type == "success" and rp.patch.edits]

    if verbose:
        print(f"    [aggregate] failure={len(failure)} success={len(success)}")

    fail_sys = load_prompt("merge_failure", overrides=prompt_overrides)
    suc_sys = load_prompt("merge_success", overrides=prompt_overrides)
    final_sys = load_prompt("merge_final", overrides=prompt_overrides)

    merged_fail = _hierarchical_merge(
        optimizer_client=optimizer_client,
        system_prompt=fail_sys,
        skill_content=skill_content,
        patches=failure,
        batch_size=batch_size,
        workers=workers,
        max_completion_tokens=max_completion_tokens,
        label="failure",
        verbose=verbose,
    )
    merged_suc = _hierarchical_merge(
        optimizer_client=optimizer_client,
        system_prompt=suc_sys,
        skill_content=skill_content,
        patches=success,
        batch_size=batch_size,
        workers=workers,
        max_completion_tokens=max_completion_tokens,
        label="success",
        verbose=verbose,
    )

    if not merged_fail.edits and not merged_suc.edits:
        return Patch(edits=[], reasoning="no usable patches")
    if not merged_suc.edits:
        return merged_fail
    if not merged_fail.edits:
        return merged_suc

    # Final merge with failure priority
    user = (
        f"## Current Skill\n{skill_content}\n\n"
        f"## Two pre-merged patch groups to combine\n"
        f"Group 1 (failure-driven, HIGH priority): "
        f"{len(merged_fail.edits)} edits\n"
        f"Group 2 (success-driven, lower priority): "
        f"{len(merged_suc.edits)} edits\n\n"
        + json.dumps(
            [merged_fail.to_dict(), merged_suc.to_dict()],
            ensure_ascii=False,
            indent=2,
        )
    )
    text, _usage = optimizer_client.chat(
        system=final_sys,
        user=user,
        max_completion_tokens=max_completion_tokens,
        stage="merge",
        retries=3,
    )
    obj = extract_json(text)
    if obj and isinstance(obj.get("edits"), list):
        if verbose:
            print(
                f"    [aggregate final] {len(merged_fail.edits)}+{len(merged_suc.edits)} "
                f"→ {len(obj.get('edits', []))}"
            )
        return Patch.from_dict(obj)

    # Fallback: failure first, then success
    return Patch(
        edits=merged_fail.edits + merged_suc.edits,
        reasoning="fallback: failure first, then success",
    )


__all__ = ["merge_patches"]
