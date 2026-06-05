"""Stage ① Rollout — parallel execution of the target agent on a batch.

The rollout is delegated to the user-supplied environment
(:class:`~skillrl.envs.base.SkillEnv`).  This module just provides a
thread pool that fans out per-item rollouts and aggregates results.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from skillrl.types import RolloutResult


def run_rollout(
    env,
    skill: str,
    items: list[dict],
    *,
    target_client,
    workers: int = 8,
    verbose: bool = True,
    stage_label: str = "rollout",
) -> list[RolloutResult]:
    """Run ``env.rollout_one(item, skill, target_client)`` for each item.

    Parameters
    ----------
    env:
        The :class:`~skillrl.envs.base.SkillEnv` instance.
    skill:
        Current skill document text.
    items:
        List of dataset items to evaluate.
    target_client:
        :class:`~skillrl.llm.base.BaseLLMClient` used as the (frozen)
        target agent.
    workers:
        Thread pool size.

    Returns
    -------
    list[RolloutResult]
        One per input item, in input order.
    """
    if not items:
        return []

    n = len(items)
    out: list[RolloutResult | None] = [None] * n

    def _one(idx: int, item: dict) -> tuple[int, RolloutResult]:
        try:
            res = env.rollout_one(item=item, skill=skill, target_client=target_client)
        except Exception as exc:  # noqa: BLE001
            res = RolloutResult(
                id=str(item.get("id", f"item_{idx}")),
                hard=0,
                soft=0.0,
                fail_reason=f"rollout_error: {type(exc).__name__}: {exc}",
            )
        if not isinstance(res, RolloutResult):
            res = RolloutResult.from_dict(res if isinstance(res, dict) else {"id": str(idx)})
        return idx, res

    if workers <= 1:
        for i, it in enumerate(items):
            _, r = _one(i, it)
            out[i] = r
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_one, i, it) for i, it in enumerate(items)]
            done = 0
            for fut in as_completed(futs):
                idx, r = fut.result()
                out[idx] = r
                done += 1
                if verbose and (done % max(1, n // 10) == 0 or done == n):
                    print(f"    [{stage_label}] {done}/{n}")

    return [r for r in out if r is not None]


def split_failures_successes(
    results: list[RolloutResult],
) -> tuple[list[RolloutResult], list[RolloutResult]]:
    """Partition rollout results by ``hard == 0`` vs ``hard == 1``."""
    failures = [r for r in results if not r.hard]
    successes = [r for r in results if r.hard]
    return failures, successes


__all__ = ["run_rollout", "split_failures_successes"]
