# skillrl

> **A TRL-like training library for end-to-end skill optimization of frozen LLM agents.**
> Implements the core algorithm of Microsoft **SkillOpt** ([project page](https://microsoft.github.io/SkillOpt/), [repo](https://github.com/microsoft/SkillOpt)) as a clean, modular Python package — designed to grow into the *TRL of skill / prompt-space optimization*.

---

## 1. What is this?

Modern LLM agents are usually improved either by **fine-tuning weights** (expensive, opaque) or by **hand-tweaking prompts** (cheap, brittle, ad-hoc).

**SkillOpt** proposes a third path: treat a **natural-language *skill document*** (a markdown file of guidelines, heuristics, do/don'ts) as the *trainable state*. Both the **target LLM** (the agent that uses the skill) and the **optimizer LLM** (the model that critiques and rewrites the skill) stay **frozen**. Gradient descent is replaced by a textual analogue:

| SGD on weights                  | SkillOpt on skill text                          |
|---------------------------------|-------------------------------------------------|
| Forward pass                    | **Rollout**: target agent runs the skill on a batch |
| Backward pass (∂L/∂θ)           | **Reflect**: optimizer LLM analyses success/failure → candidate edits |
| Gradient accumulation           | **Aggregate**: hierarchical merge → one coherent patch |
| Gradient clipping / `learning_rate` | **Select**: rank edits, keep top-`L` (the *edit budget*) |
| `optimizer.step()`              | **Update**: deterministically apply edits to the skill doc |
| Validation                      | **Evaluate**: hold-out gate — accept iff strictly better |

**`skillrl` packages this 6-stage pipeline as a TRL-style library**, so you can write:

```python
trainer = SkillOptTrainer(config=cfg, env=env, optimizer_client=..., target_client=...)
summary = trainer.train()
```

…just like you'd write `PPOTrainer(...).train()` in 🤗 TRL.

---

## 2. Why TRL-like?

| 🤗 TRL                       | skillrl                                  |
|-----------------------------|------------------------------------------|
| `PPOConfig` (dataclass)     | `SkillOptConfig` (dataclass)             |
| `PPOTrainer.train()`        | `SkillOptTrainer.train()`                |
| Reward model                | `SkillEnv` (`rollout_one` returns `hard`/`soft`/`fail_reason`) |
| Policy model (trainable)    | **Skill document** (markdown, trainable) |
| Reference / value model     | Frozen **target_client**                 |
| Optimizer (Adam)            | Frozen **optimizer_client** + edit budget scheduler |
| Learning rate               | `edit_budget` (max edits per step) + `lr_scheduler` (constant/linear/cosine) |
| Gradient clipping           | LLM-based ranking — keeps top-`L` edits  |
| Validation reward           | `selection_split` gate (`hard` / `soft` / `mixed`) |

---

## 3. Installation

```bash
# editable install from this repo
pip install -e .

# or with dev extras (pytest)
pip install -e .[dev]
```

Requirements: Python ≥ 3.10, `openai>=1.40.0`. Any OpenAI-compatible endpoint (vLLM / Together / Azure / Moonshot / DeepSeek / …) works out of the box.

---

## 4. Quick start

A minimal end-to-end example using the bundled `SimpleQAEnv`:

```python
from skillrl import SkillOptConfig, SkillOptTrainer
from skillrl.envs.qa import SimpleQAEnv
from skillrl.llm.openai_client import OpenAIChatClient

# 1. Data
train = [
    {"id": "1", "question": "Capital of France?",      "answers": ["Paris"]},
    {"id": "2", "question": "Largest ocean on Earth?", "answers": ["Pacific Ocean", "Pacific"]},
    # ... 30+ items recommended
]
val  = [{"id": "v1", "question": "Capital of Japan?", "answers": ["Tokyo"]}]
test = [{"id": "t1", "question": "Capital of Italy?", "answers": ["Rome"]}]

env = SimpleQAEnv(train_items=train, val_items=val, test_items=test)

# 2. Backends (optimizer_client = strong; target_client = the agent under training)
optimizer = OpenAIChatClient(model="gpt-4o")           # critic / rewriter
target    = OpenAIChatClient(model="gpt-4o-mini")      # the frozen agent

# 3. Config (paper-default protocol)
cfg = SkillOptConfig(
    num_epochs=2,
    batch_size=8,
    minibatch_size=4,
    edit_budget=4,
    lr_scheduler="cosine",
    gate_metric="hard",
    out_root="outputs/qa_demo",
)

# 4. Train
trainer = SkillOptTrainer(
    config=cfg, env=env,
    optimizer_client=optimizer, target_client=target,
    initial_skill="You are a concise QA assistant. Answer in one short phrase.",
)
summary = trainer.train()
print(summary["best_selection_score"], summary["test_hard"])
```

A runnable version lives at [`examples/train_qa.py`](examples/train_qa.py).

After training, the output directory contains everything you need to inspect the run:

```
outputs/qa_demo/
├── config.json                 # resolved config
├── best_skill.md               # all-time best skill (deploy this)
├── current_skill.md            # last accepted skill
├── history.json                # per-step records
├── runtime_state.json          # for auto-resume
├── summary.json                # final report
├── skills/skill_v0001.md ...   # per-step snapshots
├── steps/step_0000/
│   ├── rollout_results.json
│   ├── raw_patches.json
│   ├── merged_patch.json
│   ├── ranked_patch.json
│   ├── candidate_skill.md
│   ├── edit_apply_report.json
│   ├── selection_eval/        # validation rollouts on this candidate
│   └── step_record.json
├── test_eval_baseline/
└── test_eval_best/
```

If you re-launch with the same `out_root`, training **auto-resumes** from the last completed step.

---

## 5. The 6-stage pipeline

```
   ┌──────────────────────────────────────────────────────────────┐
   │                   one optimization step                      │
   │                                                              │
   │  current_skill.md                                            │
   │         │                                                    │
   │   ① ROLLOUT     env.rollout_one(item, skill, target_client)  │
   │         │  (parallel, n=batch_size)                          │
   │         ▼                                                    │
   │   trajectories  →  hard / soft / fail_reason                 │
   │         │                                                    │
   │   ② REFLECT     analyse failure & success minibatches        │
   │         │  optimizer_client → JSON {reasoning, edits}        │
   │         ▼                                                    │
   │   raw_patches  (failure-tagged + success-tagged)             │
   │         │                                                    │
   │   ③ AGGREGATE   hierarchical merge, failure-first            │
   │         │  optimizer_client → one coherent patch             │
   │         ▼                                                    │
   │   merged_patch                                               │
   │         │                                                    │
   │   ④ SELECT      LLM ranks edits, keep top-L (edit_budget)    │
   │         │  ≈ "gradient clipping" in text space               │
   │         ▼                                                    │
   │   ranked_patch                                               │
   │         │                                                    │
   │   ⑤ UPDATE      apply_patch(skill, ranked_patch)             │
   │         │  deterministic, append/insert/replace/delete       │
   │         ▼                                                    │
   │   candidate_skill.md                                         │
   │         │                                                    │
   │   ⑥ EVALUATE    rollout on selection_split → gate            │
   │         │  accept iff strictly better than current_score     │
   │         ▼                                                    │
   │   if accept:   current_skill := candidate                    │
   │                if also > best_score: best_skill := candidate │
   │   else:        keep current_skill                            │
   │                                                              │
   └──────────────────────────────────────────────────────────────┘
```

**Edit budget = textual learning rate.** The cap on edits applied per step is decayed (constant / linear / cosine) over the entire training horizon, exactly as the SkillOpt paper does.

**Validation gate** is strict: candidates must *strictly* beat `current_score`. A separate `best_skill` is tracked in parallel, so the artifact you ship is always the all-time best.

---

## 6. Library structure

```
skillrl/
├── __init__.py            # public exports
├── config.py              # SkillOptConfig (dataclass)
├── types.py               # Edit / Patch / RawPatch / RolloutResult / GateResult
├── trainer.py             # SkillOptTrainer — the main loop
│
├── core/
│   ├── editor.py          # apply_edit / apply_patch (5-Update)
│   ├── scheduler.py       # constant / linear / cosine edit-budget schedulers
│   ├── gate.py            # validation gate (hard / soft / mixed)
│   └── utils.py           # extract_json, compute_score, skill_hash
│
├── llm/
│   ├── base.py            # BaseLLMClient interface
│   └── openai_client.py   # OpenAI / Azure / OpenAI-compatible
│
├── pipeline/
│   ├── rollout.py         # 1-Rollout (parallel)
│   ├── reflect.py         # 2-Reflect (failure / success minibatches)
│   ├── aggregate.py       # 3-Aggregate (hierarchical merge, failure-first)
│   └── select.py          # 4-Select (LLM rank + top-L clip)
│
├── prompts/               # bundled markdown prompt templates
│   ├── analyst_error.md
│   ├── analyst_success.md
│   ├── merge_failure.md
│   ├── merge_success.md
│   ├── merge_final.md
│   └── ranking.md
│
└── envs/
    ├── base.py            # SkillEnv abstract class
    └── qa.py              # SimpleQAEnv (reference implementation)
```

---

## 7. Writing your own environment

To train a skill on your task, subclass `SkillEnv`:

```python
from skillrl.envs.base import SkillEnv
from skillrl.types import RolloutResult

class MyEnv(SkillEnv):
    name = "my_env"

    def get_initial_skill(self) -> str:
        return "You are an expert XYZ agent..."

    def get_items(self, split: str) -> list[dict]:
        return self._splits[split]            # train / val / test

    def rollout_one(self, *, item, skill, target_client) -> RolloutResult:
        # 1) build the conversation; the *skill* is typically the system prompt.
        # 2) call target_client.chat(...) one or more times (multi-turn allowed).
        # 3) score the outcome:  hard ∈ {0,1},  soft ∈ [0,1].
        # 4) return RolloutResult(...).
        ...
```

That's it — drop it into `SkillOptTrainer` and you're training.

> **Tip.** For multi-turn / tool-using agents, return the full `conversation` list and a meaningful `fail_reason`. The Reflect stage uses both to localise *why* the skill failed and *what* to change.

---

## 8. Customising prompts

All optimizer-LLM prompts live in `skillrl/prompts/*.md`. Override any of them per-trainer without modifying the package:

```python
trainer = SkillOptTrainer(
    config=cfg, env=env,
    optimizer_client=opt, target_client=tgt,
    prompt_overrides={
        "analyst_error": open("my_prompts/analyst_error.md").read(),
        "ranking":       open("my_prompts/ranking.md").read(),
    },
)
```

Available keys: `analyst_error`, `analyst_success`, `merge_failure`, `merge_success`, `merge_final`, `ranking`.

---

## 9. Reproducibility & observability

* **Determinism.** The same `seed`, `batch_size`, `minibatch_size`, dataset and backends produce the same minibatch shuffles and analyst groupings.
* **Auto-resume.** Re-running with the same `out_root` skips already-completed steps (rebuilds the selection cache from `history.json`).
* **Per-step artifacts.** Every stage's input/output is dumped — easy to diff between steps and reproduce any single step locally.
* **Selection cache.** Identical candidate skills (by `skill_hash`) reuse cached selection-split scores — saves a *lot* of money on long runs.

---

## 10. What's NOT in 1.0 (yet)

`skillrl 1.0` ships the **core algorithm** as faithfully as possible. The following SkillOpt features are intentionally deferred to future minor releases:

* `slow_update` (skill momentum / EMA over accepted skills)
* `meta_skill` (a meta-document guiding *how* to edit the skill)
* Autonomous LR (online edit-budget tuning)
* Gradient accumulation across steps
* `rewrite` / `full_rewrite_minibatch` update modes
* Codex / Claude-Code / Qwen / MiniMax execution backends
* Ray-based distributed rollouts
* WebUI

PRs welcome.

---

## 11. License

MIT. See [LICENSE](LICENSE).
