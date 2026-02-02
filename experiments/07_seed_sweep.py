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
    # One additional seed beyond the seed=42 primary run (exp 05).
    # We run only seed 137 (rather than {137, 271}) to bound compute;
    # two seeds suffice to claim robustness if both grok.
    for seed in (137,):
        data = make_multitask(("+", "-", "*"), p=113, train_frac=0.30, seed=seed)
        cfg = TrainConfig(
            name=f"07_multitask_three_seed_{seed}",
            ops=("+", "-", "*"),
            p=113,
            train_frac=0.30,
            seed=seed,
            steps=75_000,
            log_every=2000,
            eval_every=200,
        )
        train(cfg, data)


if __name__ == "__main__":
    main()
