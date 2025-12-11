"""Single-task addition baseline — replication of Nanda 2023.

Trains a 1-layer transformer on (a + b) mod 113 with a 30/70 train/test
split and weight_decay = 1.0. Expected behaviour: train loss collapses
within a few hundred steps, then test loss stays ~random-guess for ~20k
steps before dropping abruptly to near-zero.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.tasks import make_task
from src.train import TrainConfig, train


def main() -> None:
    data = make_task("+", p=113, train_frac=0.30, seed=42)
    cfg = TrainConfig(
        name="01_baseline_addition",
        ops=("+",),
        p=113,
        train_frac=0.30,
        seed=42,
        steps=30_000,
        log_every=2000,
        eval_every=200,
    )
    train(cfg, data)


if __name__ == "__main__":
    main()
