"""End-to-end demo: train a QA skill on a tiny built-in dataset.

Usage
-----

1.  Install the package:

        pip install -e .

2.  Set your API key (any OpenAI-compatible endpoint works):

        export OPENAI_API_KEY=sk-...
        # optional, for a non-OpenAI endpoint:
        # export OPENAI_BASE_URL=https://your-endpoint/v1

3.  Run:

        python examples/train_qa.py
        # or with a different model:
        python examples/train_qa.py --target-model gpt-4o-mini --optimizer-model gpt-4o

The trained best-skill is written to:
    outputs/qa_demo/best_skill.md
"""
from __future__ import annotations

import argparse
import os

from skillrl import SkillOptConfig, SkillOptTrainer
from skillrl.envs.qa import SimpleQAEnv
from skillrl.llm.openai_client import OpenAIChatClient


# ─────────────────────────────────────────────────────────────────────────
# Tiny built-in QA dataset (deliberately simple — the goal is to verify
# that the 6-stage pipeline runs end-to-end, not to set a benchmark).
# ─────────────────────────────────────────────────────────────────────────

TRAIN_ITEMS = [
    {"id": "t01", "question": "What is the capital of France?",            "answers": ["Paris"],                            "task_type": "geography"},
    {"id": "t02", "question": "What is the capital of Japan?",             "answers": ["Tokyo"],                            "task_type": "geography"},
    {"id": "t03", "question": "What is the capital of Australia?",         "answers": ["Canberra"],                         "task_type": "geography"},
    {"id": "t04", "question": "What is the capital of Canada?",            "answers": ["Ottawa"],                           "task_type": "geography"},
    {"id": "t05", "question": "What is the capital of Brazil?",            "answers": ["Brasilia", "Brasília"],             "task_type": "geography"},
    {"id": "t06", "question": "Who wrote the play 'Hamlet'?",              "answers": ["William Shakespeare", "Shakespeare"], "task_type": "literature"},
    {"id": "t07", "question": "Who wrote 'Pride and Prejudice'?",          "answers": ["Jane Austen", "Austen"],            "task_type": "literature"},
    {"id": "t08", "question": "Who wrote 'War and Peace'?",                "answers": ["Leo Tolstoy", "Tolstoy"],           "task_type": "literature"},
    {"id": "t09", "question": "Who wrote 'One Hundred Years of Solitude'?","answers": ["Gabriel Garcia Marquez", "García Márquez"], "task_type": "literature"},
    {"id": "t10", "question": "What is 12 multiplied by 12?",              "answers": ["144"],                              "task_type": "math"},
    {"id": "t11", "question": "What is 25 squared?",                       "answers": ["625"],                              "task_type": "math"},
    {"id": "t12", "question": "What is the square root of 169?",          "answers": ["13"],                               "task_type": "math"},
    {"id": "t13", "question": "What is 7 factorial (7!)?",                 "answers": ["5040"],                             "task_type": "math"},
    {"id": "t14", "question": "What is the chemical symbol for gold?",     "answers": ["Au"],                               "task_type": "science"},
    {"id": "t15", "question": "What is the chemical symbol for silver?",   "answers": ["Ag"],                               "task_type": "science"},
    {"id": "t16", "question": "What gas do plants absorb during photosynthesis?", "answers": ["Carbon dioxide", "CO2", "CO₂"], "task_type": "science"},
    {"id": "t17", "question": "Which planet is known as the Red Planet?",  "answers": ["Mars"],                             "task_type": "science"},
    {"id": "t18", "question": "Who painted the Mona Lisa?",                "answers": ["Leonardo da Vinci", "da Vinci"],    "task_type": "art"},
    {"id": "t19", "question": "Who composed 'The Four Seasons'?",          "answers": ["Antonio Vivaldi", "Vivaldi"],       "task_type": "art"},
    {"id": "t20", "question": "In which year did World War II end?",       "answers": ["1945"],                             "task_type": "history"},
]

VAL_ITEMS = [
    {"id": "v01", "question": "What is the capital of Italy?",       "answers": ["Rome"],          "task_type": "geography"},
    {"id": "v02", "question": "Who wrote 'Romeo and Juliet'?",       "answers": ["William Shakespeare", "Shakespeare"], "task_type": "literature"},
    {"id": "v03", "question": "What is 15 multiplied by 8?",         "answers": ["120"],           "task_type": "math"},
    {"id": "v04", "question": "What is the chemical symbol for iron?", "answers": ["Fe"],          "task_type": "science"},
    {"id": "v05", "question": "Who painted 'The Starry Night'?",     "answers": ["Vincent van Gogh", "van Gogh"], "task_type": "art"},
    {"id": "v06", "question": "In which year did the Berlin Wall fall?", "answers": ["1989"],     "task_type": "history"},
]

TEST_ITEMS = [
    {"id": "te01", "question": "What is the capital of Spain?",        "answers": ["Madrid"],         "task_type": "geography"},
    {"id": "te02", "question": "Who wrote '1984'?",                    "answers": ["George Orwell", "Orwell"], "task_type": "literature"},
    {"id": "te03", "question": "What is 9 cubed?",                     "answers": ["729"],            "task_type": "math"},
    {"id": "te04", "question": "What is the chemical symbol for sodium?", "answers": ["Na"],         "task_type": "science"},
    {"id": "te05", "question": "Who sculpted 'David'?",                "answers": ["Michelangelo"],   "task_type": "art"},
    {"id": "te06", "question": "In which year did humans first land on the Moon?", "answers": ["1969"], "task_type": "history"},
]


INITIAL_SKILL = """\
You are a concise question-answering assistant.

Answer the user's question with the shortest correct response possible —
ideally a single word, name, number, or short phrase. Do not add
explanations, prefixes such as "The answer is", or trailing punctuation.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="skillrl QA demo")
    parser.add_argument("--target-model",    default=os.getenv("SKILLRL_TARGET_MODEL",    "gpt-4o-mini"))
    parser.add_argument("--optimizer-model", default=os.getenv("SKILLRL_OPTIMIZER_MODEL", "gpt-4o-mini"))
    parser.add_argument("--out-root",        default="outputs/qa_demo")
    parser.add_argument("--num-epochs",      type=int, default=2)
    parser.add_argument("--batch-size",      type=int, default=8)
    parser.add_argument("--minibatch-size",  type=int, default=4)
    parser.add_argument("--edit-budget",     type=int, default=4)
    parser.add_argument("--workers",         type=int, default=4)
    parser.add_argument("--gate-metric",     default="hard", choices=["hard", "soft", "mixed"])
    parser.add_argument("--lr-scheduler",    default="cosine", choices=["constant", "linear", "cosine"])
    args = parser.parse_args()

    if not (os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")):
        raise SystemExit(
            "Please set OPENAI_API_KEY (or AZURE_OPENAI_API_KEY) in your "
            "environment before running this example."
        )

    # 1. Environment ───────────────────────────────────────────────────
    env = SimpleQAEnv(
        train_items=TRAIN_ITEMS,
        val_items=VAL_ITEMS,
        test_items=TEST_ITEMS,
        initial_skill=INITIAL_SKILL,
    )

    # 2. Backends ──────────────────────────────────────────────────────
    base_url = os.getenv("OPENAI_BASE_URL")  # e.g. for vLLM / Together / Moonshot
    target_client = OpenAIChatClient(model=args.target_model,    base_url=base_url)
    optimizer_client = OpenAIChatClient(model=args.optimizer_model, base_url=base_url)

    # 3. Config ────────────────────────────────────────────────────────
    cfg = SkillOptConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        minibatch_size=args.minibatch_size,
        merge_batch_size=4,
        edit_budget=args.edit_budget,
        min_edit_budget=2,
        lr_scheduler=args.lr_scheduler,
        gate_metric=args.gate_metric,
        workers=args.workers,
        out_root=args.out_root,
        seed=42,
        verbose=True,
    )

    # 4. Train ─────────────────────────────────────────────────────────
    trainer = SkillOptTrainer(
        config=cfg,
        env=env,
        optimizer_client=optimizer_client,
        target_client=target_client,
        initial_skill=INITIAL_SKILL,
    )
    summary = trainer.train()

    # 5. Report ────────────────────────────────────────────────────────
    print("\n=== summary ===")
    print(f"steps        : {summary['total_steps']}")
    print(f"accepts      : {summary['total_accepts']}")
    print(f"rejects      : {summary['total_rejects']}")
    print(f"skips        : {summary['total_skips']}")
    print(f"best step    : {summary['best_step']}")
    print(f"best (val)   : {summary['best_selection_score']:.4f}")
    if summary["test_hard"] is not None:
        print(
            f"test (hard)  : baseline={summary['baseline_test_hard']:.4f}  "
            f"best={summary['test_hard']:.4f}  "
            f"delta={summary['test_delta_hard']:+.4f}"
        )
    print(f"\nbest skill saved to: {os.path.join(args.out_root, 'best_skill.md')}")


if __name__ == "__main__":
    main()
