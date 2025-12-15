"""Three-seed robustness sweep on the multi-task-three configuration.

Reruns experiment 05 with seeds 137 and 271 (seed 42 is the primary run).
Confirms that the multi-task grokking dynamics are not an artefact of one
particular initialisation.
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
    for seed in (137, 271):
        data = make_multitask(("+", "-", "*"), p=113, train_frac=0.30, seed=seed)
        cfg = TrainConfig(
            name=f"07_multitask_three_seed_{seed}",
            ops=("+", "-", "*"),
            p=113,
            train_frac=0.30,
            seed=seed,
            steps=25_000,
            log_every=2000,
            eval_every=200,
        )
        train(cfg, data)


if __name__ == "__main__":
    main()
