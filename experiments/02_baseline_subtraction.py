"""Single-task subtraction baseline.

Subtraction is the same group operation as addition (the additive group of
Z/pZ), so the dynamics should look qualitatively the same as the addition
baseline. The Fourier analysis should likewise discover features on the
additive Z/pZ group.
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
    data = make_task("-", p=113, train_frac=0.30, seed=42)
    cfg = TrainConfig(
        name="02_baseline_subtraction",
        ops=("-",),
        p=113,
        train_frac=0.30,
        seed=42,
        steps=15_000,
        log_every=2000,
        eval_every=200,
    )
    train(cfg, data)


if __name__ == "__main__":
    main()
