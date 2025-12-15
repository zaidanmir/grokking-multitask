"""Multi-task experiment 2: addition + subtraction + multiplication jointly.

Centre of the project's research contribution. Two of the three operations
live on the additive group of Z/pZ; multiplication lives on the
multiplicative group (Z/pZ)^*. The experiment asks whether one network
can solve all three within the same residual stream — and if so, what the
internal representation looks like.
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
    data = make_multitask(("+", "-", "*"), p=113, train_frac=0.30, seed=42)
    cfg = TrainConfig(
        name="05_multitask_three",
        ops=("+", "-", "*"),
        p=113,
        train_frac=0.30,
        seed=42,
        steps=25_000,
        log_every=2000,
        eval_every=200,
    )
    train(cfg, data)


if __name__ == "__main__":
    main()
