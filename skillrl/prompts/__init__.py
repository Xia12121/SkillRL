"""Default system prompts for the skillrl Reflect / Aggregate / Select stages.

These templates are environment-agnostic and produce strict-JSON output
to maximise robustness.  Users may override any prompt by passing
``prompt_overrides={"analyst_error": "..."}`` to the trainer.
"""
from __future__ import annotations

import importlib.resources as ir
from functools import lru_cache


_KNOWN_PROMPTS = {
    "analyst_error",
    "analyst_success",
    "merge_failure",
    "merge_success",
    "merge_final",
    "ranking",
}


@lru_cache(maxsize=None)
def _read_prompt(name: str) -> str:
    if name not in _KNOWN_PROMPTS:
        raise KeyError(f"Unknown prompt {name!r}; known: {sorted(_KNOWN_PROMPTS)}")
    pkg = "skillrl.prompts"
    try:
        return ir.files(pkg).joinpath(f"{name}.md").read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        raise FileNotFoundError(
            f"Could not read built-in prompt template {name!r}: {exc}"
        ) from exc


def load_prompt(name: str, overrides: dict[str, str] | None = None) -> str:
    """Load a prompt template, with optional user override.

    Parameters
    ----------
    name:
        Template name (without ``.md``).  See :data:`_KNOWN_PROMPTS`.
    overrides:
        Optional mapping ``{name: full_text}`` to substitute the bundled
        template.  Useful for domain-specific tuning without forking
        the package.
    """
    if overrides and name in overrides:
        return overrides[name]
    return _read_prompt(name)


__all__ = ["load_prompt"]
