"""Environment / dataset abstractions used by the trainer.

A ``SkillEnv`` defines:

* a train / val / test split,
* how to roll out a single item with a given skill (calling the target
  LLM via the supplied client),
* how to score the rollout (``hard``: 0/1 exact-match, ``soft``: [0,1]
  partial credit).

Bring-your-own-env: subclass :class:`SkillEnv` and pass the instance to
:class:`~skillrl.SkillOptTrainer`.  See :class:`~skillrl.envs.qa.SimpleQAEnv`
for a complete reference implementation.
"""

from skillrl.envs.base import SkillEnv
from skillrl.envs.qa import SimpleQAEnv

__all__ = ["SkillEnv", "SimpleQAEnv"]
