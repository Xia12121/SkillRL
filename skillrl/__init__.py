"""skillrl — A TRL-like training library for end-to-end skill optimization.

This package implements the core algorithm of Microsoft SkillOpt
(https://microsoft.github.io/SkillOpt/) as a clean, modular, TRL-style
training library.  The trainable state is a natural-language *skill
document*; both the optimizer and the target LLM stay frozen.

Quick start
-----------
>>> from skillrl import SkillOptConfig, SkillOptTrainer
>>> from skillrl.envs.qa import SimpleQAEnv
>>> from skillrl.llm.openai_client import OpenAIChatClient
>>>
>>> cfg = SkillOptConfig(num_epochs=2, batch_size=8, edit_budget=4)
>>> env = SimpleQAEnv(train_items=[...], val_items=[...], test_items=[...])
>>> client = OpenAIChatClient(model="gpt-4o-mini")
>>> trainer = SkillOptTrainer(
...     config=cfg, env=env,
...     optimizer_client=client, target_client=client,
...     initial_skill="You are a helpful assistant.",
... )
>>> summary = trainer.train()
>>> print(summary["test_hard"])
"""

from skillrl.config import SkillOptConfig
from skillrl.types import (
    Edit,
    Patch,
    RolloutResult,
    GateResult,
    GateAction,
    RawPatch,
)
from skillrl.trainer import SkillOptTrainer

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "SkillOptConfig",
    "SkillOptTrainer",
    "Edit",
    "Patch",
    "RolloutResult",
    "GateResult",
    "GateAction",
    "RawPatch",
]
