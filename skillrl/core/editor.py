"""Skill-document edit application — the Update stage (⑤).

Analogous to ``optimizer.step()`` in neural network training: a ranked
:class:`~skillrl.types.Patch` is applied sequentially onto the current
skill document, producing a candidate skill that is then validated by
the gate.
"""
from __future__ import annotations

from typing import Any

from skillrl.types import Edit, Patch


def _edit_fields(edit: Edit | dict) -> tuple[str, str, str]:
    """Extract ``(op, content, target)`` from an ``Edit`` or plain dict."""
    if isinstance(edit, Edit):
        return edit.op, (edit.content or "").strip(), edit.target or ""
    return (
        str(edit.get("op", "") or ""),
        str(edit.get("content", "") or "").strip(),
        str(edit.get("target", "") or ""),
    )


def _apply_edit_with_report(skill: str, edit: Edit | dict) -> tuple[str, dict]:
    op, content, target = _edit_fields(edit)
    report: dict[str, Any] = {
        "op": op,
        "target": target[:200],
        "content_preview": content[:200],
        "status": "unknown",
    }

    if op == "append":
        report["status"] = "applied_append"
        return skill.rstrip() + "\n\n" + content + "\n", report

    if op == "insert_after":
        if not target or target not in skill:
            report["status"] = "applied_insert_after_fallback_append"
            return skill.rstrip() + "\n\n" + content + "\n", report
        idx = skill.index(target) + len(target)
        newline = skill.find("\n", idx)
        insert_at = newline + 1 if newline != -1 else len(skill)
        report["status"] = "applied_insert_after"
        return skill[:insert_at] + "\n" + content + "\n" + skill[insert_at:], report

    if op == "replace":
        if not target:
            report["status"] = "skipped_replace_missing_target"
            return skill, report
        if target not in skill:
            report["status"] = "skipped_replace_target_not_found"
            return skill, report
        report["status"] = "applied_replace"
        return skill.replace(target, content, 1), report

    if op == "delete":
        if not target:
            report["status"] = "skipped_delete_missing_target"
            return skill, report
        if target not in skill:
            report["status"] = "skipped_delete_target_not_found"
            return skill, report
        report["status"] = "applied_delete"
        return skill.replace(target, "", 1), report

    report["status"] = "skipped_unknown_op"
    return skill, report


def apply_edit(skill: str, edit: Edit | dict) -> str:
    """Apply a single edit operation to the skill document."""
    updated, _ = _apply_edit_with_report(skill, edit)
    return updated


def apply_patch_with_report(
    skill: str,
    patch: Patch | dict,
) -> tuple[str, list[dict]]:
    """Apply a patch and return a per-edit report for observability."""
    if isinstance(patch, Patch):
        edits: list[Edit | dict] = list(patch.edits)
    else:
        edits = list(patch.get("edits", []) or [])

    reports: list[dict] = []
    for idx, edit in enumerate(edits, 1):
        try:
            skill, report = _apply_edit_with_report(skill, edit)
            report["index"] = idx
        except Exception as exc:  # noqa: BLE001
            report = {
                "index": idx,
                "op": "",
                "target": "",
                "content_preview": "",
                "status": "error",
                "error": str(exc),
            }
        reports.append(report)
    return skill, reports


def apply_patch(skill: str, patch: Patch | dict) -> str:
    """Apply a patch (list of edits) to the skill document sequentially."""
    updated, _ = apply_patch_with_report(skill, patch)
    return updated
