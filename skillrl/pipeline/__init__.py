"""skillrl pipeline stages — the LLM-driven training loop primitives.

* :mod:`skillrl.pipeline.rollout`   — Stage ① Rollout (parallel evaluation)
* :mod:`skillrl.pipeline.reflect`   — Stage ② Reflect (analyst → patches)
* :mod:`skillrl.pipeline.aggregate` — Stage ③ Aggregate (hierarchical merge)
* :mod:`skillrl.pipeline.select`    — Stage ④ Select (rank & clip)

Stage ⑤ Update lives in :mod:`skillrl.core.editor`; Stage ⑥ Evaluate is
implemented inline by the trainer (it is just rollout + gate).
"""

from skillrl.pipeline.rollout import run_rollout
from skillrl.pipeline.reflect import run_minibatch_reflect
from skillrl.pipeline.aggregate import merge_patches
from skillrl.pipeline.select import rank_and_select

__all__ = [
    "run_rollout",
    "run_minibatch_reflect",
    "merge_patches",
    "rank_and_select",
]
