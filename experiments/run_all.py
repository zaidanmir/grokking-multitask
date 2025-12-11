"""Run every experiment in sequence, then regenerate all figures.

Usage::

    PYTHONPATH=. python experiments/run_all.py [--skip-existing]

If ``--skip-existing`` is passed, runs whose ``runs/<name>/final.pt``
already exists are skipped — useful for resuming after a partial run.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENTS = [
    "01_baseline_addition",
    "02_baseline_subtraction",
    "03_baseline_multiplication",
    "04_multitask_two",
    "05_multitask_three",
    "06_curriculum",
    "07_seed_sweep",
    "08_robustness_sweeps",
]


def _module_run_names(short: str) -> list[str]:
    """Best-effort guess at the run name(s) a module produces."""
    if short == "07_seed_sweep":
        return ["07_multitask_three_seed_137", "07_multitask_three_seed_271"]
    if short == "06_curriculum":
        return ["06_curriculum_stage_a_add", "06_curriculum_stage_b_multi"]
    if short == "08_robustness_sweeps":
        return [
            "08_robustness_p59_tf30",
            "08_robustness_p199_tf30",
            "08_robustness_p113_tf40",
            "08_robustness_p113_tf50",
        ]
    return [short]


def _all_done(short: str) -> bool:
    runs = PROJECT_ROOT / "runs"
    return all((runs / r / "final.pt").exists() for r in _module_run_names(short))


def _run_module(short: str) -> None:
    """Load and run an experiment script by file path (handles digit-prefix names)."""
    path = PROJECT_ROOT / "experiments" / f"{short}.py"
    spec = importlib.util.spec_from_file_location(f"_exp_{short}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--only", nargs="*", default=None,
        help="run only the listed module short names (e.g. 01_baseline_addition)",
    )
    args = parser.parse_args()

    selected = EXPERIMENTS
    if args.only:
        selected = [m for m in EXPERIMENTS if m in args.only]

    overall_t0 = time.time()
    for short in selected:
        if args.skip_existing and _all_done(short):
            print(f"\n=== SKIP {short} (already complete) ===\n", flush=True)
            continue
        print(f"\n=== RUN {short} ===\n", flush=True)
        t0 = time.time()
        _run_module(short)
        print(f"\n=== DONE {short} in {time.time() - t0:.1f}s ===\n", flush=True)
    print(f"\nTotal time: {time.time() - overall_t0:.1f}s")


if __name__ == "__main__":
    main()
