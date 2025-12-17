"""Quick exploratory look at the addition replication run.

Run as ``PYTHONPATH=. python notebooks/01_replication_eda.py``. Loads
the saved metric history, prints the headline numbers, and shows a
two-panel matplotlib figure (train/test loss + train/test accuracy).
This is the script form of what would normally be a Jupyter notebook
cell-by-cell exploration.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.eval import find_grok_step
from src.train import load_history


def main() -> None:
    h = load_history(PROJECT_ROOT / "runs" / "01_baseline_addition")

    print("=== single-task addition replication ===")
    print(f"  total eval points: {len(h['step'])}")
    print(f"  final step: {h['step'][-1]:,}")
    print(f"  final train loss: {h['train_loss'][-1]:.3e}")
    print(f"  final test  loss: {h['test_loss'][-1]:.3e}")
    print(f"  final train acc:  {h['train_acc'][-1]:.4f}")
    print(f"  final test  acc:  {h['test_acc'][-1]:.4f}")
    for thr in (0.50, 0.95, 0.99):
        s = find_grok_step(h["step"].tolist(), h["test_acc"].tolist(), threshold=thr)
        print(f"  test acc >= {thr:.2f}: step {s}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(h["step"], h["train_loss"], label="train", color="#1f77b4")
    axes[0].plot(h["step"], h["test_loss"], label="test", color="#ff7f0e")
    axes[0].set_yscale("log")
    axes[0].set_xscale("symlog", linthresh=100)
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss")
    axes[0].legend(frameon=False)

    axes[1].plot(h["step"], h["train_acc"], label="train", color="#1f77b4")
    axes[1].plot(h["step"], h["test_acc"], label="test", color="#ff7f0e")
    axes[1].axhline(0.95, color="grey", ls=":", lw=0.7)
    axes[1].set_xscale("symlog", linthresh=100)
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("accuracy")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
