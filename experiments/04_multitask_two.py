"""Multi-task experiment 1: addition + subtraction jointly.

Both operations live on the additive group of Z/pZ, so the natural
hypothesis is that the model can solve them with a shared set of additive
Fourier features (cosines and sines indexed by frequencies on Z/pZ). The
question is whether the multi-task model groks at the same step count as
the single-task model, and whether the discovered feature set is the same.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.tasks import make_multitask
from src.train import TrainConfig, train


def main() -> None:
    data = make_multitask(("+", "-"), p=113, train_frac=0.30, seed=42)
    cfg = TrainConfig(
        name="04_multitask_two",
        ops=("+", "-"),
        p=113,
        train_frac=0.30,
        seed=42,
        steps=20_000,
        log_every=2000,
        eval_every=200,
    )
    train(cfg, data)


if __name__ == "__main__":
    main()
