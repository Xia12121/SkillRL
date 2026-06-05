"""skillrl core algorithms — pure-functional building blocks.

This subpackage hosts the *non-LLM* primitives used by the trainer:

* :mod:`skillrl.core.editor`     — apply edits / patches to a skill doc.
* :mod:`skillrl.core.scheduler`  — edit-budget (textual LR) schedulers.
* :mod:`skillrl.core.gate`       — validation gating (hard/soft/mixed).
* :mod:`skillrl.core.utils`      — JSON extraction, scoring, hashing.
"""

from skillrl.core.editor import apply_edit, apply_patch, apply_patch_with_report
from skillrl.core.scheduler import (
    LRScheduler,
    ConstantScheduler,
    LinearScheduler,
    CosineScheduler,
    build_scheduler,
)
from skillrl.core.gate import evaluate_gate, select_gate_score
from skillrl.core.utils import compute_score, extract_json, skill_hash

__all__ = [
    "apply_edit",
    "apply_patch",
    "apply_patch_with_report",
    "LRScheduler",
    "ConstantScheduler",
    "LinearScheduler",
    "CosineScheduler",
    "build_scheduler",
    "evaluate_gate",
    "select_gate_score",
    "compute_score",
    "extract_json",
    "skill_hash",
]
