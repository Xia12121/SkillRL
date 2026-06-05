"""Offline unit tests for skillrl core building blocks.

These tests exercise only the deterministic, non-LLM parts of the
library, so they run instantly and require no API keys / network.

Run with:

    pytest -q tests/
"""
from __future__ import annotations

import json

import pytest

from skillrl import (
    Edit,
    Patch,
    RolloutResult,
    SkillOptConfig,
)
from skillrl.core.editor import apply_edit, apply_patch, apply_patch_with_report
from skillrl.core.gate import evaluate_gate, select_gate_score
from skillrl.core.scheduler import (
    ConstantScheduler,
    CosineScheduler,
    LinearScheduler,
    build_scheduler,
)
from skillrl.core.utils import compute_score, extract_json, skill_hash


# ─────────────────────────────────────────────────────────────────────────
# Editor — the Update stage
# ─────────────────────────────────────────────────────────────────────────


class TestEditor:
    def test_append_adds_block_at_the_end(self):
        skill = "Header line.\n"
        out = apply_edit(skill, Edit(op="append", content="Rule X."))
        assert out.endswith("Rule X.\n")
        assert "Header line." in out

    def test_replace_when_target_exists(self):
        skill = "be concise.\nbe accurate.\n"
        out = apply_edit(
            skill, Edit(op="replace", target="be concise.", content="Be very concise.")
        )
        assert "Be very concise." in out
        assert "be concise." not in out

    def test_replace_skipped_when_target_missing(self):
        skill = "alpha\n"
        out = apply_edit(skill, Edit(op="replace", target="missing", content="x"))
        assert out == skill  # unchanged

    def test_delete_removes_target(self):
        skill = "A\nB\nC\n"
        out = apply_edit(skill, Edit(op="delete", target="B\n"))
        assert "B" not in out
        assert "A" in out and "C" in out

    def test_insert_after_inserts_below_target_line(self):
        skill = "## Heading\nfirst line.\nsecond line.\n"
        out = apply_edit(
            skill,
            Edit(op="insert_after", target="## Heading", content="new line."),
        )
        # New content sits between the heading and the first line.
        assert out.index("new line.") > out.index("## Heading")
        assert out.index("new line.") < out.index("first line.")

    def test_insert_after_falls_back_to_append_when_target_missing(self):
        skill = "alpha\n"
        out = apply_edit(
            skill, Edit(op="insert_after", target="ghost", content="beta")
        )
        assert "beta" in out
        assert out.startswith("alpha")

    def test_unknown_op_is_a_noop(self):
        skill = "alpha\n"
        out = apply_edit(skill, {"op": "frobnicate", "content": "x"})
        assert out == skill

    def test_apply_patch_runs_edits_in_order(self):
        skill = "be concise.\n"
        patch = Patch(
            edits=[
                Edit(op="replace", target="be concise.", content="Be very concise."),
                Edit(op="append", content="Always cite the source."),
            ]
        )
        out = apply_patch(skill, patch)
        assert "Be very concise." in out
        assert "Always cite the source." in out

    def test_apply_patch_with_report_classifies_each_edit(self):
        skill = "alpha\n"
        patch = Patch(
            edits=[
                Edit(op="append", content="A"),
                Edit(op="replace", target="missing", content="X"),
                Edit(op="delete", target="missing"),
            ]
        )
        _, reports = apply_patch_with_report(skill, patch)
        statuses = [r["status"] for r in reports]
        assert statuses[0].startswith("applied")
        assert statuses[1].startswith("skipped")
        assert statuses[2].startswith("skipped")


# ─────────────────────────────────────────────────────────────────────────
# Schedulers — the textual learning rate
# ─────────────────────────────────────────────────────────────────────────


class TestScheduler:
    def test_constant_returns_max_lr_every_step(self):
        sch = ConstantScheduler(max_lr=4, min_lr=2, total_steps=8)
        values = [sch.step() for _ in range(5)]
        assert all(v == 4 for v in values)

    def test_linear_decays_monotonically(self):
        sch = LinearScheduler(max_lr=10, min_lr=2, total_steps=10)
        values = [sch.step() for _ in range(10)]
        # First value < max_lr (already decaying), last value >= min_lr.
        assert values[0] >= values[-1]
        assert values[-1] >= 2
        assert max(values) <= 10

    def test_cosine_starts_high_and_ends_at_min(self):
        sch = CosineScheduler(max_lr=8, min_lr=2, total_steps=10)
        values = [sch.step() for _ in range(10)]
        assert values[0] >= values[-1]
        assert values[-1] == 2

    def test_state_dict_round_trip(self):
        sch = CosineScheduler(max_lr=8, min_lr=2, total_steps=10)
        for _ in range(3):
            sch.step()
        state = sch.state_dict()

        sch2 = CosineScheduler(max_lr=8, min_lr=2, total_steps=10)
        sch2.load_state_dict(state)
        # Both should yield the same next value.
        assert sch.step() == sch2.step()

    def test_build_scheduler_dispatches_correctly(self):
        assert isinstance(build_scheduler("constant", 4, 2, 5), ConstantScheduler)
        assert isinstance(build_scheduler("linear", 4, 2, 5), LinearScheduler)
        assert isinstance(build_scheduler("cosine", 4, 2, 5), CosineScheduler)
        with pytest.raises(ValueError):
            build_scheduler("unknown", 4, 2, 5)


# ─────────────────────────────────────────────────────────────────────────
# Gate — the validation gate
# ─────────────────────────────────────────────────────────────────────────


class TestGate:
    def test_select_gate_score_modes(self):
        assert select_gate_score(0.7, 0.5, "hard") == pytest.approx(0.7)
        assert select_gate_score(0.7, 0.5, "soft") == pytest.approx(0.5)
        assert select_gate_score(0.7, 0.5, "mixed", 0.5) == pytest.approx(0.6)
        with pytest.raises(ValueError):
            select_gate_score(0.5, 0.5, "garbage")  # type: ignore[arg-type]

    def _gate(self, **kw):
        defaults = dict(
            candidate_skill="cand",
            cand_hard=0.0,
            current_skill="cur",
            current_score=0.0,
            best_skill="best",
            best_score=0.0,
            best_step=0,
            global_step=1,
            cand_soft=0.0,
            metric="hard",
        )
        defaults.update(kw)
        return evaluate_gate(**defaults)

    def test_strict_improvement_over_best_promotes_both(self):
        g = self._gate(cand_hard=0.9, current_score=0.5, best_score=0.7, best_step=0,
                       global_step=5)
        assert g.action == "accept_new_best"
        assert g.current_skill == "cand"
        assert g.best_skill == "cand"
        assert g.best_step == 5

    def test_improvement_over_current_only_keeps_best(self):
        g = self._gate(cand_hard=0.6, current_score=0.5, best_score=0.7, best_step=2)
        assert g.action == "accept"
        assert g.current_skill == "cand"
        assert g.best_skill == "best"
        assert g.best_step == 2

    def test_no_strict_improvement_rejects(self):
        # Equal scores must NOT pass — the gate is strict.
        g = self._gate(cand_hard=0.5, current_score=0.5, best_score=0.5, best_step=3)
        assert g.action == "reject"
        assert g.current_skill == "cur"
        assert g.best_skill == "best"


# ─────────────────────────────────────────────────────────────────────────
# Utils — JSON extraction / scoring / hashing
# ─────────────────────────────────────────────────────────────────────────


class TestUtils:
    def test_extract_json_raw(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_extract_json_fenced(self):
        text = "Sure, here it is:\n```json\n{\"k\": [1, 2]}\n```\nDone."
        assert extract_json(text) == {"k": [1, 2]}

    def test_extract_json_brace_slice(self):
        text = "Some preface {\"x\": \"y\"} and tail."
        assert extract_json(text) == {"x": "y"}

    def test_extract_json_invalid_returns_none(self):
        assert extract_json("not json at all") is None
        assert extract_json("") is None
        assert extract_json(None) is None  # type: ignore[arg-type]

    def test_compute_score_handles_dataclass_and_dict(self):
        results = [
            RolloutResult(id="1", hard=1, soft=1.0),
            RolloutResult(id="2", hard=0, soft=0.5),
            {"hard": 1, "soft": 0.8},
        ]
        h, s = compute_score(results)
        assert h == pytest.approx(2 / 3)
        assert s == pytest.approx((1.0 + 0.5 + 0.8) / 3)

    def test_compute_score_empty(self):
        assert compute_score([]) == (0.0, 0.0)

    def test_skill_hash_is_deterministic_and_short(self):
        a = skill_hash("hello")
        b = skill_hash("hello")
        c = skill_hash("hello world")
        assert a == b
        assert a != c
        assert len(a) == 16


# ─────────────────────────────────────────────────────────────────────────
# Config — sanity & validation
# ─────────────────────────────────────────────────────────────────────────


class TestConfig:
    def test_default_protocol_matches_paper(self):
        cfg = SkillOptConfig()
        assert cfg.num_epochs == 4
        assert cfg.batch_size == 40
        assert cfg.minibatch_size == 8
        assert cfg.edit_budget == 4
        assert cfg.lr_scheduler == "cosine"
        assert cfg.gate_metric == "hard"

    def test_to_dict_round_trips_through_json(self):
        cfg = SkillOptConfig(num_epochs=1, batch_size=2, minibatch_size=2,
                             merge_batch_size=2, edit_budget=2, min_edit_budget=1)
        s = json.dumps(cfg.to_dict())
        loaded = json.loads(s)
        assert loaded["num_epochs"] == 1
        assert loaded["edit_budget"] == 2

    @pytest.mark.parametrize(
        "bad",
        [
            {"batch_size": 0},
            {"minibatch_size": 0},
            {"merge_batch_size": 1},
            {"edit_budget": 0},
            {"min_edit_budget": 0},
            {"min_edit_budget": 999},  # > edit_budget
            {"lr_scheduler": "bogus"},
            {"gate_metric": "bogus"},
            {"gate_mixed_weight": 1.5},
        ],
    )
    def test_invalid_config_raises(self, bad):
        with pytest.raises(ValueError):
            SkillOptConfig(**bad)


# ─────────────────────────────────────────────────────────────────────────
# Types — round-trip serialisation
# ─────────────────────────────────────────────────────────────────────────


class TestTypes:
    def test_edit_round_trip(self):
        e = Edit(op="replace", content="new", target="old", source_type="failure",
                 support_count=3, merge_level=2)
        d = e.to_dict()
        assert Edit.from_dict(d).to_dict() == d

    def test_patch_round_trip(self):
        p = Patch(
            edits=[Edit(op="append", content="X"), Edit(op="delete", target="Y")],
            reasoning="why",
        )
        d = p.to_dict()
        p2 = Patch.from_dict(d)
        assert p2.reasoning == "why"
        assert len(p2.edits) == 2
        assert p2.edits[0].op == "append"

    def test_rollout_result_extras_preserved(self):
        r = RolloutResult(id="x", hard=1, soft=1.0, extras={"custom": 42})
        d = r.to_dict()
        assert d["custom"] == 42
        r2 = RolloutResult.from_dict(d)
        assert r2.extras.get("custom") == 42
