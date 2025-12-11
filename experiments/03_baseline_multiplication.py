"""Single-task multiplication baseline.

Multiplication mod a prime p operates on the cyclic group (Z/pZ)^* of order
p-1, which has different Fourier-transform structure than the additive
group. Multiplication is empirically harder than addition under the same
hyperparameters: this script uses a longer schedule (50k steps) to give
the optimiser room to grok.
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
    data = make_task("*", p=113, train_frac=0.30, seed=42)
    cfg = TrainConfig(
        name="03_baseline_multiplication",
        ops=("*",),
        p=113,
        train_frac=0.30,
        seed=42,
        steps=50_000,
        log_every=2000,
        eval_every=200,
    )
    train(cfg, data)


if __name__ == "__main__":
    main()
