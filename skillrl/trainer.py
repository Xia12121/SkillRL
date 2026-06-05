"""skillrl Trainer — the SkillOpt main loop, packaged TRL-style.

The trainer orchestrates the 6-stage pipeline:

    ① Rollout    target agent runs the current skill on a train minibatch
    ② Reflect    optimizer LLM analyses trajectories → candidate patches
    ③ Aggregate  hierarchical merge → one coherent patch
    ④ Select     LLM ranks edits, keeps top-L (clip)
    ⑤ Update     deterministically apply the patch → candidate skill
    ⑥ Evaluate   roll out the candidate on the held-out selection split,
                 accept iff strictly better than running ``current_score``

A separate global ``best_skill`` is tracked so that the artifact you
deploy at the end is always the all-time-best.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from typing import Any

from skillrl.config import SkillOptConfig
from skillrl.core.editor import apply_patch_with_report
from skillrl.core.gate import evaluate_gate, select_gate_score
from skillrl.core.scheduler import build_scheduler
from skillrl.core.utils import compute_score, skill_hash
from skillrl.envs.base import SkillEnv
from skillrl.llm.base import BaseLLMClient
from skillrl.pipeline.aggregate import merge_patches
from skillrl.pipeline.reflect import run_minibatch_reflect
from skillrl.pipeline.rollout import run_rollout
from skillrl.pipeline.select import rank_and_select
from skillrl.types import Patch, RolloutResult


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _read_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _write_text(path: str, content: str) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj: Any) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _shuffle_in_chunks(items: list[Any], chunk: int, seed: int) -> list[list[Any]]:
    """Deterministically shuffle ``items`` and slice into chunks of ``chunk``.

    The last chunk may be shorter.
    """
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    return [shuffled[i : i + chunk] for i in range(0, len(shuffled), chunk)]


def _per_task_type_buckets(
    results: list[RolloutResult],
    task_types: list[str],
) -> dict[str, dict]:
    buckets: dict[str, dict] = {
        t: {"total": 0, "hard": 0, "soft": 0.0} for t in task_types + ["overall"]
    }
    for r in results:
        tt = r.task_type or "other"
        if tt not in buckets:
            buckets[tt] = {"total": 0, "hard": 0, "soft": 0.0}
        buckets[tt]["total"] += 1
        buckets[tt]["hard"] += int(r.hard)
        buckets[tt]["soft"] += float(r.soft)
        buckets["overall"]["total"] += 1
        buckets["overall"]["hard"] += int(r.hard)
        buckets["overall"]["soft"] += float(r.soft)
    return buckets


# ─────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────


class SkillOptTrainer:
    """End-to-end skill optimizer (TRL-style API).

    Parameters
    ----------
    config:
        :class:`~skillrl.config.SkillOptConfig` instance.
    env:
        The user's :class:`~skillrl.envs.base.SkillEnv`.
    optimizer_client:
        Backend used to call the optimizer LLM (Reflect / Aggregate /
        Select).  Strong models recommended.
    target_client:
        Backend used to call the (frozen) target agent during rollouts.
        Can point to the same or a weaker model than ``optimizer_client``.
    initial_skill:
        Optional seed skill text.  If ``None``, the env's
        :meth:`~skillrl.envs.base.SkillEnv.get_initial_skill` is used.
    prompt_overrides:
        Optional dict mapping prompt name → full text to override the
        bundled defaults.
    """

    def __init__(
        self,
        *,
        config: SkillOptConfig,
        env: SkillEnv,
        optimizer_client: BaseLLMClient,
        target_client: BaseLLMClient,
        initial_skill: str | None = None,
        prompt_overrides: dict[str, str] | None = None,
    ) -> None:
        self.cfg = config
        self.env = env
        self.optimizer_client = optimizer_client
        self.target_client = target_client
        self.prompt_overrides = prompt_overrides or {}

        self.initial_skill = (
            initial_skill if initial_skill is not None else env.get_initial_skill()
        )

        _ensure_dir(self.cfg.out_root)

        # ── Persisted state ──────────────────────────────────────────
        self.history: list[dict] = []
        self.runtime_state: dict[str, Any] = {}
        self.sel_cache: dict[str, tuple[float, float]] = {}

        self.current_skill: str = self.initial_skill
        self.best_skill: str = self.initial_skill
        self.current_score: float = -1.0
        self.best_score: float = -1.0
        self.best_step: int = 0
        self.global_step: int = 0

        # Total step count is determined once batches are known
        self._total_steps: int = 0
        self._steps_per_epoch: int = 0
        self._scheduler = None  # type: ignore[assignment]

    # ── Persistence ──────────────────────────────────────────────────

    def _save_skill_snapshot(self, step: int, content: str) -> None:
        if not self.cfg.save_every_step:
            return
        path = os.path.join(self.cfg.out_root, "skills", f"skill_v{step:04d}.md")
        _write_text(path, content)

    def _save_history(self) -> None:
        _write_json(os.path.join(self.cfg.out_root, "history.json"), self.history)

    def _persist_runtime_state(self) -> None:
        self.runtime_state = {
            "last_completed_step": self.global_step,
            "current_score": self.current_score,
            "best_score": self.best_score,
            "best_step": self.best_step,
            "scheduler_state": (
                self._scheduler.state_dict() if self._scheduler is not None else {}
            ),
        }
        _write_json(
            os.path.join(self.cfg.out_root, "runtime_state.json"),
            self.runtime_state,
        )
        _write_text(os.path.join(self.cfg.out_root, "best_skill.md"), self.best_skill)
        _write_text(
            os.path.join(self.cfg.out_root, "current_skill.md"), self.current_skill
        )

    def _try_resume(self) -> bool:
        runtime_path = os.path.join(self.cfg.out_root, "runtime_state.json")
        history_path = os.path.join(self.cfg.out_root, "history.json")
        if not (os.path.exists(runtime_path) and os.path.exists(history_path)):
            return False
        try:
            self.runtime_state = _read_json(runtime_path)
            self.history = _read_json(history_path)
        except Exception:
            return False

        last_step = int(self.runtime_state.get("last_completed_step", 0) or 0)
        if last_step <= 0:
            return False

        self.global_step = last_step
        self.current_score = float(self.runtime_state.get("current_score", -1.0))
        self.best_score = float(self.runtime_state.get("best_score", self.current_score))
        self.best_step = int(self.runtime_state.get("best_step", 0))

        cur_path = os.path.join(self.cfg.out_root, "current_skill.md")
        best_path = os.path.join(self.cfg.out_root, "best_skill.md")
        if os.path.exists(cur_path):
            self.current_skill = _read_text(cur_path)
        if os.path.exists(best_path):
            self.best_skill = _read_text(best_path)

        # Rebuild selection cache from history
        for rec in self.history:
            ch = rec.get("candidate_hash", "")
            if ch and rec.get("selection_hard") is not None:
                self.sel_cache[ch] = (
                    float(rec["selection_hard"]),
                    float(rec.get("selection_soft", 0.0) or 0.0),
                )

        if self.cfg.verbose:
            print(
                f"  [resume] from step {self.global_step + 1}  "
                f"current={self.current_score:.4f} best={self.best_score:.4f}"
            )
        return True

    # ── Selection / test evaluation ─────────────────────────────────

    def _eval_on_split(
        self,
        skill: str,
        split: str,
        out_subdir: str,
    ) -> tuple[float, float, list[RolloutResult]]:
        items = self.env.get_items(split)
        if not items:
            return 0.0, 0.0, []
        results = run_rollout(
            env=self.env,
            skill=skill,
            items=items,
            target_client=self.target_client,
            workers=self.cfg.workers,
            verbose=self.cfg.verbose,
            stage_label=f"{split}_eval",
        )
        h, s = compute_score(results)
        eval_dir = os.path.join(self.cfg.out_root, out_subdir)
        _ensure_dir(eval_dir)
        _write_json(
            os.path.join(eval_dir, "results.json"),
            [r.to_dict() for r in results],
        )
        _write_json(
            os.path.join(eval_dir, "summary.json"),
            {"hard": h, "soft": s, "n": len(results)},
        )
        return h, s, results

    def _ensure_baseline_evaluated(self) -> None:
        """Evaluate the initial skill on the selection split if not yet done."""
        if self.current_score >= 0:
            return
        if self.cfg.verbose:
            print("\n  [BASELINE] evaluating initial skill on selection split")
        hard, soft, _ = self._eval_on_split(
            self.initial_skill,
            self.cfg.selection_split,
            "baseline_selection_eval",
        )
        score = select_gate_score(
            hard, soft, self.cfg.gate_metric, self.cfg.gate_mixed_weight
        )
        self.current_score = score
        self.best_score = score
        self.sel_cache[skill_hash(self.initial_skill)] = (hard, soft)
        if self.cfg.verbose:
            print(
                f"  [baseline] selection hard={hard:.4f} soft={soft:.4f} "
                f"gate[{self.cfg.gate_metric}]={score:.4f}"
            )

    # ── Per-step pipeline ───────────────────────────────────────────

    def _do_step(self, *, epoch: int, step_in_epoch: int, batch_items: list[dict]) -> dict:
        """Run all six stages for one optimization step. Returns step record."""
        step_dir = os.path.join(
            self.cfg.out_root, "steps", f"step_{self.global_step:04d}"
        )
        _ensure_dir(step_dir)
        t0 = time.time()
        rec: dict[str, Any] = {
            "step": self.global_step,
            "epoch": epoch,
            "step_in_epoch": step_in_epoch,
            "timing": {},
        }

        if self.cfg.verbose:
            print(
                f"\n  [STEP {self.global_step}/{self._total_steps}] "
                f"epoch={epoch} batch_size={len(batch_items)} "
                + "=" * 30
            )

        # ── ① Rollout ─────────────────────────────────────────────────
        t = time.time()
        rollout_results = run_rollout(
            env=self.env,
            skill=self.current_skill,
            items=batch_items,
            target_client=self.target_client,
            workers=self.cfg.workers,
            verbose=self.cfg.verbose,
            stage_label="1/6 rollout",
        )
        r_hard, r_soft = compute_score(rollout_results)
        rec["timing"]["rollout_s"] = round(time.time() - t, 2)
        rec["rollout_hard"] = round(r_hard, 6)
        rec["rollout_soft"] = round(r_soft, 6)
        rec["rollout_n"] = len(rollout_results)
        if self.cfg.verbose:
            print(f"    [1/6 done] hard={r_hard:.4f} soft={r_soft:.4f}")

        _write_json(
            os.path.join(step_dir, "rollout_results.json"),
            [r.to_dict() for r in rollout_results],
        )

        # ── ② Reflect ────────────────────────────────────────────────
        t = time.time()
        edit_budget_for_analyst = max(self.cfg.edit_budget, self.cfg.min_edit_budget)
        raw_patches = run_minibatch_reflect(
            rollout_results=rollout_results,
            current_skill=self.current_skill,
            optimizer_client=self.optimizer_client,
            minibatch_size=self.cfg.minibatch_size,
            edit_budget=edit_budget_for_analyst,
            failure_only=self.cfg.failure_only,
            workers=self.cfg.workers,
            seed=self.cfg.seed + self.global_step,
            max_completion_tokens=self.cfg.optimizer_max_completion_tokens,
            prompt_overrides=self.prompt_overrides,
            verbose=self.cfg.verbose,
        )
        rec["timing"]["reflect_s"] = round(time.time() - t, 2)
        rec["n_failure_patches"] = sum(
            1 for rp in raw_patches if rp.source_type == "failure"
        )
        rec["n_success_patches"] = sum(
            1 for rp in raw_patches if rp.source_type == "success"
        )

        _write_json(
            os.path.join(step_dir, "raw_patches.json"),
            [rp.to_dict() for rp in raw_patches],
        )

        if not raw_patches:
            rec["action"] = "skip_no_patches"
            rec["current_score"] = self.current_score
            rec["best_score"] = self.best_score
            rec["best_step"] = self.best_step
            rec["wall_time_s"] = round(time.time() - t0, 2)
            if self.cfg.verbose:
                print("    [skip] no patches produced — skill unchanged")
            return rec

        # ── ③ Aggregate ──────────────────────────────────────────────
        t = time.time()
        merged: Patch = merge_patches(
            optimizer_client=self.optimizer_client,
            skill_content=self.current_skill,
            raw_patches=raw_patches,
            batch_size=self.cfg.merge_batch_size,
            workers=self.cfg.workers,
            max_completion_tokens=self.cfg.optimizer_max_completion_tokens,
            prompt_overrides=self.prompt_overrides,
            verbose=self.cfg.verbose,
        )
        rec["timing"]["aggregate_s"] = round(time.time() - t, 2)
        rec["n_edits_merged"] = len(merged.edits)
        _write_json(os.path.join(step_dir, "merged_patch.json"), merged.to_dict())
        if not merged.edits:
            rec["action"] = "skip_no_patches"
            rec["current_score"] = self.current_score
            rec["best_score"] = self.best_score
            rec["best_step"] = self.best_step
            rec["wall_time_s"] = round(time.time() - t0, 2)
            if self.cfg.verbose:
                print("    [skip] merged patch is empty — skill unchanged")
            return rec
        if self.cfg.verbose:
            print(f"    [3/6 done] merged {len(merged.edits)} edits")

        # ── ④ Select ─────────────────────────────────────────────────
        t = time.time()
        edit_budget = self._scheduler.step()
        ranked: Patch = rank_and_select(
            optimizer_client=self.optimizer_client,
            skill_content=self.current_skill,
            patch=merged,
            max_edits=edit_budget,
            max_completion_tokens=2048,
            prompt_overrides=self.prompt_overrides,
            verbose=self.cfg.verbose,
        )
        rec["timing"]["select_s"] = round(time.time() - t, 2)
        rec["edit_budget"] = edit_budget
        rec["n_edits_ranked"] = len(ranked.edits)
        _write_json(os.path.join(step_dir, "ranked_patch.json"), ranked.to_dict())

        # ── ⑤ Update ─────────────────────────────────────────────────
        t = time.time()
        candidate_skill, apply_report = apply_patch_with_report(
            self.current_skill, ranked
        )
        rec["timing"]["update_s"] = round(time.time() - t, 2)
        rec["candidate_skill_len"] = len(candidate_skill)
        cand_hash = skill_hash(candidate_skill)
        rec["candidate_hash"] = cand_hash
        rec["apply_report"] = {
            "total": len(apply_report),
            "applied": sum(
                1 for r in apply_report if str(r.get("status", "")).startswith("applied")
            ),
            "skipped": sum(
                1 for r in apply_report if str(r.get("status", "")).startswith("skipped")
            ),
            "errors": sum(1 for r in apply_report if r.get("status") == "error"),
        }
        _write_text(os.path.join(step_dir, "candidate_skill.md"), candidate_skill)
        _write_json(os.path.join(step_dir, "edit_apply_report.json"), apply_report)
        if self.cfg.verbose:
            print(
                f"    [5/6 UPDATE] "
                f"skill_len {len(self.current_skill)} → {len(candidate_skill)}  "
                f"(applied={rec['apply_report']['applied']}, "
                f"skipped={rec['apply_report']['skipped']})"
            )

        # ── ⑥ Evaluate (validation gate) ─────────────────────────────
        t = time.time()
        if cand_hash in self.sel_cache:
            cand_hard, cand_soft = self.sel_cache[cand_hash]
            if self.cfg.verbose:
                print(f"    [6/6 EVALUATE] cache hit {cand_hash}: hard={cand_hard:.4f}")
        else:
            cand_hard, cand_soft, _ = self._eval_on_split(
                candidate_skill,
                self.cfg.selection_split,
                f"steps/step_{self.global_step:04d}/selection_eval",
            )
            self.sel_cache[cand_hash] = (cand_hard, cand_soft)

        gate = evaluate_gate(
            candidate_skill=candidate_skill,
            cand_hard=cand_hard,
            current_skill=self.current_skill,
            current_score=self.current_score,
            best_skill=self.best_skill,
            best_score=self.best_score,
            best_step=self.best_step,
            global_step=self.global_step,
            cand_soft=cand_soft,
            metric=self.cfg.gate_metric,
            mixed_weight=self.cfg.gate_mixed_weight,
        )
        cand_gate_score = select_gate_score(
            cand_hard, cand_soft, self.cfg.gate_metric, self.cfg.gate_mixed_weight
        )

        rec["selection_hard"] = cand_hard
        rec["selection_soft"] = cand_soft
        rec["candidate_gate_score"] = cand_gate_score
        rec["action"] = gate.action
        rec["gate_metric"] = self.cfg.gate_metric
        rec["timing"]["evaluate_s"] = round(time.time() - t, 2)

        prev_current = self.current_score
        prev_best = self.best_score
        self.current_skill = gate.current_skill
        self.current_score = gate.current_score
        self.best_skill = gate.best_skill
        self.best_score = gate.best_score
        self.best_step = gate.best_step

        if self.cfg.verbose:
            tag = (
                f"{self.cfg.gate_metric}={cand_gate_score:.4f}"
                if self.cfg.gate_metric != "mixed"
                else (
                    f"mixed[w={self.cfg.gate_mixed_weight}]={cand_gate_score:.4f} "
                    f"(hard={cand_hard:.4f} soft={cand_soft:.4f})"
                )
            )
            if gate.action == "accept_new_best":
                print(
                    f"    [6/6 EVALUATE] ACCEPT (new best) {tag} > prev_best={prev_best:.4f}"
                )
            elif gate.action == "accept":
                print(
                    f"    [6/6 EVALUATE] ACCEPT {tag} > current={prev_current:.4f}"
                )
            else:
                print(
                    f"    [6/6 EVALUATE] REJECT {tag} <= current={self.current_score:.4f}"
                )

        rec["current_score"] = self.current_score
        rec["best_score"] = self.best_score
        rec["best_step"] = self.best_step
        rec["skill_len"] = len(self.current_skill)
        rec["wall_time_s"] = round(time.time() - t0, 2)
        return rec

    # ── Public entry point ──────────────────────────────────────────

    def train(self) -> dict:
        """Run the full training loop. Returns a summary dict."""
        cfg = self.cfg
        env = self.env

        _ensure_dir(cfg.out_root)
        # Persist resolved config
        _write_json(os.path.join(cfg.out_root, "config.json"), cfg.to_dict())

        train_items = env.get_items("train")
        if not train_items:
            raise ValueError("Training split is empty — cannot train.")
        train_size = len(train_items)
        steps_per_epoch = max(1, math.ceil(train_size / cfg.batch_size))
        self._steps_per_epoch = steps_per_epoch
        self._total_steps = steps_per_epoch * cfg.num_epochs

        self._scheduler = build_scheduler(
            mode=cfg.lr_scheduler,
            max_lr=cfg.edit_budget,
            min_lr=cfg.min_edit_budget,
            total_steps=self._total_steps,
        )

        if cfg.verbose:
            print("\n" + "=" * 60)
            print(f"  skillrl train · env={env.name} · steps={self._total_steps}")
            print(f"  epochs={cfg.num_epochs}  steps/epoch={steps_per_epoch}  "
                  f"batch_size={cfg.batch_size}  train_size={train_size}")
            print(f"  lr_scheduler={cfg.lr_scheduler}  edit_budget={cfg.edit_budget} "
                  f"min={cfg.min_edit_budget}")
            print(f"  gate_metric={cfg.gate_metric}  selection_split={cfg.selection_split}")
            print("=" * 60)

        # Resume?
        resumed = self._try_resume()
        if resumed:
            self._scheduler.load_state_dict(
                self.runtime_state.get("scheduler_state", {})
            )

        # Save initial snapshot (step 0) and baseline-evaluate
        self._save_skill_snapshot(0, self.initial_skill)
        self._ensure_baseline_evaluated()
        self._persist_runtime_state()

        # ── Main loop ──────────────────────────────────────────────
        t_loop = time.time()
        for epoch in range(1, cfg.num_epochs + 1):
            epoch_seed = cfg.seed + epoch * 1000
            chunks = _shuffle_in_chunks(train_items, cfg.batch_size, epoch_seed)
            # Pad/trim to exactly steps_per_epoch chunks (deterministic)
            if len(chunks) < steps_per_epoch:
                chunks += [chunks[0]] * (steps_per_epoch - len(chunks))
            chunks = chunks[:steps_per_epoch]

            if cfg.verbose:
                print(f"\n  [EPOCH {epoch}/{cfg.num_epochs}] "
                      f"steps_in_epoch={steps_per_epoch}  seed={epoch_seed}")

            for step_in_epoch, batch_items in enumerate(chunks):
                self.global_step += 1
                if resumed and self.global_step <= int(
                    self.runtime_state.get("last_completed_step", 0) or 0
                ):
                    continue

                rec = self._do_step(
                    epoch=epoch,
                    step_in_epoch=step_in_epoch,
                    batch_items=batch_items,
                )
                self.history.append(rec)
                self._save_skill_snapshot(self.global_step, self.current_skill)
                self._save_history()
                self._persist_runtime_state()

                step_dir = os.path.join(
                    cfg.out_root, "steps", f"step_{self.global_step:04d}"
                )
                _write_json(os.path.join(step_dir, "step_record.json"), rec)

                if cfg.verbose:
                    timing = rec.get("timing", {})
                    print(
                        f"\n  [STEP {self.global_step} done] "
                        f"action={rec.get('action')} "
                        f"current={self.current_score:.4f} "
                        f"best={self.best_score:.4f} "
                        f"dt={rec.get('wall_time_s', 0)}s\n"
                        f"    timing: rollout={timing.get('rollout_s', 0)}s "
                        f"reflect={timing.get('reflect_s', 0)}s "
                        f"aggregate={timing.get('aggregate_s', 0)}s "
                        f"select={timing.get('select_s', 0)}s "
                        f"evaluate={timing.get('evaluate_s', 0)}s"
                    )

        # ── Final test evaluation ──────────────────────────────────
        baseline_test_hard = baseline_test_soft = None
        test_hard = test_soft = None
        if cfg.eval_test:
            if cfg.verbose:
                print("\n" + "=" * 60)
                print("  Final evaluation on test split")
                print("=" * 60)
            baseline_test_hard, baseline_test_soft, base_results = self._eval_on_split(
                self.initial_skill, cfg.test_split, "test_eval_baseline",
            )
            test_hard, test_soft, best_results = self._eval_on_split(
                self.best_skill, cfg.test_split, "test_eval_best",
            )
            task_types = env.get_task_types()
            if task_types:
                base_buckets = _per_task_type_buckets(base_results, task_types)
                best_buckets = _per_task_type_buckets(best_results, task_types)
                _write_json(
                    os.path.join(cfg.out_root, "test_eval_baseline", "buckets.json"),
                    base_buckets,
                )
                _write_json(
                    os.path.join(cfg.out_root, "test_eval_best", "buckets.json"),
                    best_buckets,
                )

            if cfg.verbose:
                print(
                    f"\n  [test] baseline hard={baseline_test_hard:.4f}  "
                    f"best hard={test_hard:.4f}  "
                    f"delta={(test_hard - baseline_test_hard):+.4f}"
                )

        # ── Summary ────────────────────────────────────────────────
        n_accept = sum(1 for h in self.history if "accept" in str(h.get("action", "")))
        n_reject = sum(1 for h in self.history if h.get("action") == "reject")
        n_skip = sum(
            1 for h in self.history if str(h.get("action", "")).startswith("skip")
        )
        summary = {
            "version": "skillrl-1.0.0",
            "config": cfg.to_dict(),
            "env": env.name,
            "total_steps": len(self.history),
            "total_accepts": n_accept,
            "total_rejects": n_reject,
            "total_skips": n_skip,
            "best_step": self.best_step,
            "best_selection_score": self.best_score,
            "current_selection_score": self.current_score,
            "baseline_test_hard": baseline_test_hard,
            "baseline_test_soft": baseline_test_soft,
            "test_hard": test_hard,
            "test_soft": test_soft,
            "test_delta_hard": (
                None if test_hard is None or baseline_test_hard is None
                else round(test_hard - baseline_test_hard, 6)
            ),
            "wall_time_s": round(time.time() - t_loop, 2),
        }
        _write_json(os.path.join(cfg.out_root, "summary.json"), summary)
        _write_text(os.path.join(cfg.out_root, "best_skill.md"), self.best_skill)

        if cfg.verbose:
            print("\n" + "=" * 60)
            print("  Final Summary")
            print("=" * 60)
            print(
                f"  steps={summary['total_steps']} "
                f"accept={n_accept} reject={n_reject} skip={n_skip}"
            )
            print(
                f"  best_score={self.best_score:.4f} "
                f"(step {self.best_step})  wall={summary['wall_time_s']:.0f}s"
            )
            if test_hard is not None:
                print(
                    f"  test_hard={test_hard:.4f} "
                    f"(baseline {baseline_test_hard:.4f}, "
                    f"delta {summary['test_delta_hard']:+.4f})"
                )
            print(f"  best skill saved to {os.path.join(cfg.out_root, 'best_skill.md')}")

        return summary
