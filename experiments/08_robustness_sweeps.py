"""Robustness sweeps over modulus p and train fraction.

Quick small-budget runs to demonstrate that the headline grokking dynamics
of the addition baseline are not an artefact of the specific p=113,
train_frac=0.30 configuration.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.tasks import make_task
from src.train import TrainConfig, train


def run_one(p: int, train_frac: float, steps: int) -> None:
    name = f"08_robustness_p{p}_tf{int(round(train_frac * 100))}"
    data = make_task("+", p=p, train_frac=train_frac, seed=42)
    cfg = TrainConfig(
        name=name,
        ops=("+",),
        p=p,
        train_frac=train_frac,
        seed=42,
        steps=steps,
        log_every=5000,
        eval_every=500,
    )
    train(cfg, data)


def main() -> None:
    # Fix train_frac=0.30, vary p.
    # p=59 with train_frac=0.30 has only ~1k train examples — too sparse
    # to grok in a small budget. Bump steps to give it room.
    run_one(p=59, train_frac=0.30, steps=40_000)
    run_one(p=199, train_frac=0.30, steps=40_000)

    # Fix p=113, vary train_frac.
    run_one(p=113, train_frac=0.40, steps=20_000)
    run_one(p=113, train_frac=0.50, steps=15_000)


if __name__ == "__main__":
    main()
