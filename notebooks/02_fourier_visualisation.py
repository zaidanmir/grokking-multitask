"""Visualise Fourier features in the trained embedding matrices.

Loads each single-task model and the multi-task model, runs the
additive- and multiplicative-group DFT on the digit embeddings, and
plots the per-frequency power spectra side-by-side. Useful to sanity-
check the analysis pipeline before committing to a paper figure.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.analysis.fourier import (
    compute_fourier_basis,
    compute_fourier_basis_multiplicative,
    identify_dominant_frequencies,
)
from src.model import ModelConfig, OneLayerTransformer

P = 113
RUNS = PROJECT_ROOT / "runs"


def _load(name: str) -> OneLayerTransformer | None:
    path = RUNS / name / "final.pt"
    if not path.exists():
        return None
    cfg = ModelConfig(p=P)
    model = OneLayerTransformer(cfg)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def main() -> None:
    runs = {
        "single-add": "01_baseline_addition",
        "single-sub": "02_baseline_subtraction",
        "single-mul": "03_baseline_multiplication",
        "multitask-3": "05_multitask_three",
    }
    fig, axes = plt.subplots(len(runs), 2, figsize=(10, 2.0 * len(runs)),
                              sharex="col")
    for r, (label, name) in enumerate(runs.items()):
        m = _load(name)
        if m is None:
            for c in (0, 1):
                axes[r, c].text(0.5, 0.5, f"no run: {name}",
                                transform=axes[r, c].transAxes, ha="center")
            continue
        W = m.W_E.weight[:P].detach().cpu().numpy()
        spec_add = compute_fourier_basis(W, P)
        spec_mul = compute_fourier_basis_multiplicative(W, P)

        for c, spec in ((0, spec_add), (1, spec_mul)):
            power = spec.freq_power.copy()
            power[0] = 0.0
            if power.max() > 0:
                power /= power.max()
            axes[r, c].bar(np.arange(len(power)), power, width=1.0,
                           color="#1f77b4" if c == 0 else "#d62728")
            axes[r, c].set_ylim(0, 1.05)
            dom = identify_dominant_frequencies(spec, fraction=0.90)
            for k in dom:
                axes[r, c].axvline(k, color="black", lw=0.4, alpha=0.4)

        axes[r, 0].set_ylabel(label, rotation=0, labelpad=40, va="center")

    axes[0, 0].set_title(r"additive basis $\mathbb{Z}/p\mathbb{Z}$")
    axes[0, 1].set_title(r"multiplicative basis $(\mathbb{Z}/p\mathbb{Z})^*$")
    axes[-1, 0].set_xlabel("frequency $k$")
    axes[-1, 1].set_xlabel("frequency $k$")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
