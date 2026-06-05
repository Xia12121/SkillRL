"""Misc utilities — JSON extraction, scoring, hashing."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable

from skillrl.types import RolloutResult


# ── Robust JSON extraction from LLM output ──────────────────────────────


_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> dict | None:
    """Best-effort extraction of a JSON object embedded in *text*.

    Tries (in order):

    1. parsing the entire string as JSON,
    2. parsing the contents of the first fenced code block,
    3. parsing the substring between the first ``{`` and last ``}``.

    Returns ``None`` on failure — callers must handle this gracefully.
    """
    if text is None:
        return None
    s = text.strip()
    if not s:
        return None

    # 1. raw JSON
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # 2. fenced block
    m = _FENCE_RE.search(s)
    if m:
        try:
            obj = json.loads(m.group(1).strip())
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

    # 3. brace slice (last-resort)
    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = s[first : last + 1]
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

    return None


# ── Aggregate rollout scoring ───────────────────────────────────────────


def compute_score(results: Iterable[Any]) -> tuple[float, float]:
    """Return ``(hard_acc, soft_acc)`` averaged over a rollout batch.

    Accepts ``RolloutResult`` instances or plain dicts (with ``hard``
    and ``soft`` keys).  Empty input yields ``(0.0, 0.0)``.
    """
    items = list(results)
    if not items:
        return 0.0, 0.0
    hard_total = 0.0
    soft_total = 0.0
    for r in items:
        if isinstance(r, RolloutResult):
            hard_total += float(r.hard)
            soft_total += float(r.soft)
        else:
            hard_total += float(r.get("hard", 0) or 0)
            soft_total += float(r.get("soft", 0.0) or 0.0)
    n = len(items)
    return hard_total / n, soft_total / n


# ── Skill content hashing (cache key for selection eval) ────────────────


def skill_hash(content: str) -> str:
    """Deterministic short hash for a skill document (cache key)."""
    h = hashlib.sha256((content or "").encode("utf-8")).hexdigest()
    return h[:16]
