"""Generate all paper figures from saved per-step metric histories.

Reads runs/<name>/history.npz and runs/<name>/final.pt produced by the
training experiments, computes any post-hoc analyses (Fourier
decomposition, ablations), and writes vector-PDF figures and LaTeX
table snippets to paper/figures/.

The figures are intentionally minimal — black-on-white, large fonts,
clean axes — for camera-ready inclusion in the manuscript.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.tasks import make_multitask, make_task
from src.analysis.ablation import ablate_all_heads, per_task_head_ablation
from src.analysis.fourier import (
    compute_fourier_basis,
    compute_fourier_basis_multiplicative,
    feature_overlap,
    identify_dominant_frequencies,
)
from src.eval import find_grok_step
from src.model import ModelConfig, OneLayerTransformer
from src.train import load_history


# -----------------------------------------------------------------------------
# Style.
# -----------------------------------------------------------------------------
mpl.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 100,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,  # embed fonts as TrueType
    }
)

OPS_DISPLAY = {"+": "add", "-": "sub", "*": "mul"}
OP_COLOURS = {"+": "#1f77b4", "-": "#2ca02c", "*": "#d62728"}

RUNS = PROJECT_ROOT / "runs"
FIG_DIR = PROJECT_ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Helpers.
# -----------------------------------------------------------------------------
def _run_dir(name: str) -> Path:
    return RUNS / name


def _has_run(name: str) -> bool:
    return (_run_dir(name) / "history.npz").exists()


def _safe_log(arr: np.ndarray) -> np.ndarray:
    return np.where(arr > 0, np.log10(np.clip(arr, 1e-12, None)), -12)


def _load_model(run_name: str, p: int = 113) -> OneLayerTransformer:
    cfg = ModelConfig(p=p)
    model = OneLayerTransformer(cfg)
    state = torch.load(_run_dir(run_name) / "final.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


# -----------------------------------------------------------------------------
# Figure 1: addition replication.
# -----------------------------------------------------------------------------
def fig1_replication() -> None:
    h = load_history(_run_dir("01_baseline_addition"))
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.plot(h["step"], h["train_loss"], color="#1f77b4", label="train loss", lw=1.5)
    ax.plot(h["step"], h["test_loss"], color="#ff7f0e", label="test loss", lw=1.5)
    ax.set_xscale("symlog", linthresh=100)
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title(r"single-task addition, $p = 113$, $\lambda = 1.0$")
    grok = find_grok_step(h["step"].tolist(), h["test_acc"].tolist())
    if grok is not None:
        ax.axvline(grok, color="grey", ls="--", lw=0.8, alpha=0.7)
        ax.text(
            grok, ax.get_ylim()[1] * 0.4, f"  grok @ {grok}",
            fontsize=9, color="grey",
        )
    ax.legend(frameon=False, loc="lower left")
    fig.savefig(FIG_DIR / "fig1_replication.pdf")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure 2: per-op single-task baselines.
# -----------------------------------------------------------------------------
def fig2_per_op_baselines() -> None:
    runs = {
        "+": "01_baseline_addition",
        "-": "02_baseline_subtraction",
        "*": "03_baseline_multiplication",
    }
    fig, axes = plt.subplots(2, 1, figsize=(5.5, 4.5), sharex=True)
    for op, name in runs.items():
        if not _has_run(name):
            continue
        h = load_history(_run_dir(name))
        c = OP_COLOURS[op]
        axes[0].plot(h["step"], h["test_acc"], color=c, label=OPS_DISPLAY[op], lw=1.5)
        axes[1].plot(h["step"], h["test_loss"], color=c, label=OPS_DISPLAY[op], lw=1.5)
    axes[0].set_xscale("symlog", linthresh=100)
    axes[1].set_xscale("symlog", linthresh=100)
    axes[1].set_yscale("log")
    axes[0].set_ylabel("test accuracy")
    axes[1].set_ylabel("test loss")
    axes[1].set_xlabel("training step")
    axes[0].axhline(0.95, color="grey", ls=":", lw=0.7, alpha=0.7)
    axes[0].set_ylim(-0.02, 1.05)
    axes[0].legend(frameon=False, ncol=3, loc="lower right")
    fig.suptitle(r"single-task baselines, $p = 113$", y=0.995)
    fig.savefig(FIG_DIR / "fig2_per_op_baselines.pdf")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure 3: multitask add+sub.
# -----------------------------------------------------------------------------
def fig3_multitask_two() -> None:
    name = "04_multitask_two"
    if not _has_run(name):
        return
    h = load_history(_run_dir(name))
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for op in ("+", "-"):
        key = f"test_acc_{op}"
        if key in h:
            ax.plot(h["step"], h[key], color=OP_COLOURS[op],
                    label=f"test acc {OPS_DISPLAY[op]}", lw=1.5)
    ax.set_xscale("symlog", linthresh=100)
    ax.set_xlabel("training step")
    ax.set_ylabel("test accuracy")
    ax.set_title(r"multitask: $\textsc{add} + \textsc{sub}$, $p = 113$")
    ax.axhline(0.95, color="grey", ls=":", lw=0.7, alpha=0.7)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(FIG_DIR / "fig3_multitask_two.pdf")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure 4: multitask all three.
# -----------------------------------------------------------------------------
def fig4_multitask_three() -> None:
    name = "05_multitask_three"
    if not _has_run(name):
        return
    h = load_history(_run_dir(name))
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for op in ("+", "-", "*"):
        key = f"test_acc_{op}"
        if key in h:
            ax.plot(h["step"], h[key], color=OP_COLOURS[op],
                    label=f"test acc {OPS_DISPLAY[op]}", lw=1.5)
    ax.set_xscale("symlog", linthresh=100)
    ax.set_xlabel("training step")
    ax.set_ylabel("test accuracy")
    ax.set_title(r"multitask: $\textsc{add}+\textsc{sub}+\textsc{mul}$, $p = 113$")
    ax.axhline(0.95, color="grey", ls=":", lw=0.7, alpha=0.7)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(FIG_DIR / "fig4_multitask_three.pdf")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure 5: seed sweep on multitask-3.
# -----------------------------------------------------------------------------
def fig5_seed_sweep() -> None:
    runs = [
        ("05_multitask_three", 42),
        ("07_multitask_three_seed_137", 137),
        ("07_multitask_three_seed_271", 271),
    ]
    histories: list[dict] = []
    for name, _ in runs:
        if _has_run(name):
            histories.append(load_history(_run_dir(name)))
    if len(histories) < 1:
        return

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0), sharey=True)
    for ax, op in zip(axes, ("+", "-", "*")):
        per_seed = []
        common_steps = histories[0]["step"]
        for h in histories:
            key = f"test_acc_{op}"
            if key not in h:
                continue
            per_seed.append(h[key])
        if not per_seed:
            continue
        # Align by step (assumes same eval cadence; truncate to shortest).
        L = min(len(s) for s in per_seed)
        steps = histories[0]["step"][:L]
        per_seed = [s[:L] for s in per_seed]
        arr = np.stack(per_seed, axis=0)
        for s in arr:
            ax.plot(steps, s, color=OP_COLOURS[op], alpha=0.45, lw=1.0)
        ax.fill_between(
            steps, arr.min(axis=0), arr.max(axis=0),
            color=OP_COLOURS[op], alpha=0.18,
        )
        ax.plot(steps, arr.mean(axis=0), color=OP_COLOURS[op], lw=1.5)
        ax.axhline(0.95, color="grey", ls=":", lw=0.7, alpha=0.7)
        ax.set_xscale("symlog", linthresh=100)
        ax.set_xlabel("training step")
        ax.set_title(OPS_DISPLAY[op])
        ax.set_ylim(-0.02, 1.05)
    axes[0].set_ylabel("test accuracy")
    fig.suptitle(r"three-seed sweep on multitask-3, $p = 113$", y=1.02)
    fig.savefig(FIG_DIR / "fig5_seed_sweep.pdf")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figures 6 / 7: Fourier spectra.
# -----------------------------------------------------------------------------
def _embedding_digits(model: OneLayerTransformer, p: int) -> np.ndarray:
    return model.W_E.weight[:p].detach().cpu().numpy()


def fig6_fourier_singletask(p: int = 113) -> None:
    runs = {
        "+": "01_baseline_addition",
        "-": "02_baseline_subtraction",
        "*": "03_baseline_multiplication",
    }
    fig, axes = plt.subplots(3, 1, figsize=(6.0, 5.0), sharex=True)
    for ax, op in zip(axes, ("+", "-", "*")):
        name = runs[op]
        if not _has_run(name):
            ax.text(0.5, 0.5, f"no run: {name}", transform=ax.transAxes, ha="center")
            continue
        model = _load_model(name, p=p)
        spec = compute_fourier_basis(_embedding_digits(model, p), p)
        # Normalise so subplots are visually comparable.
        power = spec.freq_power.copy()
        power[0] = 0.0  # drop DC
        if power.max() > 0:
            power = power / power.max()
        ax.bar(np.arange(p), power, color=OP_COLOURS[op], width=1.0)
        ax.set_ylabel(OPS_DISPLAY[op])
        ax.set_ylim(0, 1.05)
        dom = identify_dominant_frequencies(spec, fraction=0.90)
        for k in dom:
            ax.axvline(k, color="black", lw=0.4, alpha=0.5)
            ax.axvline(p - k, color="black", lw=0.4, alpha=0.5)
    axes[-1].set_xlabel(r"additive-group frequency $k \in \{0, \ldots, p-1\}$")
    fig.suptitle(r"Fourier power spectrum of digit embeddings on $\mathbb{Z}/p\mathbb{Z}$",
                 y=1.0)
    fig.savefig(FIG_DIR / "fig6_fourier_singletask.pdf")
    plt.close(fig)


def fig7_fourier_mult(p: int = 113) -> None:
    name = "03_baseline_multiplication"
    if not _has_run(name):
        return
    model = _load_model(name, p=p)
    spec_add = compute_fourier_basis(_embedding_digits(model, p), p)
    spec_mul = compute_fourier_basis_multiplicative(_embedding_digits(model, p), p)

    fig, axes = plt.subplots(2, 1, figsize=(6.0, 4.2))
    for ax, spec, label, c in (
        (axes[0], spec_add, r"additive basis $\mathbb{Z}/p\mathbb{Z}$", "#888888"),
        (axes[1], spec_mul, r"multiplicative basis $(\mathbb{Z}/p\mathbb{Z})^*$",
         OP_COLOURS["*"]),
    ):
        power = spec.freq_power.copy()
        power[0] = 0.0
        if power.max() > 0:
            power = power / power.max()
        ax.bar(np.arange(len(power)), power, color=c, width=1.0)
        ax.set_ylabel("normalised power")
        ax.set_title(label)
        ax.set_ylim(0, 1.05)
    axes[1].set_xlabel(r"frequency index")
    fig.suptitle("multiplication model: same embeddings, two bases", y=1.01)
    fig.savefig(FIG_DIR / "fig7_fourier_mult.pdf")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure 8: feature overlap.
# -----------------------------------------------------------------------------
def fig8_feature_overlap(p: int = 113) -> None:
    single_runs = {
        "+": "01_baseline_addition",
        "-": "02_baseline_subtraction",
        "*": "03_baseline_multiplication",
    }
    multi_run = "05_multitask_three"
    if not _has_run(multi_run):
        return
    multi_model = _load_model(multi_run, p=p)
    multi_W = _embedding_digits(multi_model, p)
    multi_add = compute_fourier_basis(multi_W, p)
    multi_mul = compute_fourier_basis_multiplicative(multi_W, p)

    overlaps_add = {}
    overlaps_mul = {}
    for op, name in single_runs.items():
        if not _has_run(name):
            continue
        m = _load_model(name, p=p)
        W = _embedding_digits(m, p)
        s_add = compute_fourier_basis(W, p)
        s_mul = compute_fourier_basis_multiplicative(W, p)
        overlaps_add[op] = feature_overlap(multi_add, s_add)
        overlaps_mul[op] = feature_overlap(multi_mul, s_mul)

    ops = ("+", "-", "*")
    add_vals = [overlaps_add.get(op, 0.0) for op in ops]
    mul_vals = [overlaps_mul.get(op, 0.0) for op in ops]
    x = np.arange(3)
    width = 0.38
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.bar(x - width / 2, add_vals, width, label="additive-basis overlap",
           color="#1f77b4")
    ax.bar(x + width / 2, mul_vals, width, label="multiplicative-basis overlap",
           color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels([f"vs single-{OPS_DISPLAY[op]}" for op in ops])
    ax.set_ylabel("cosine similarity")
    ax.set_ylim(0, 1.05)
    ax.set_title("feature overlap: multitask-3 vs single-task models")
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(FIG_DIR / "fig8_feature_overlap.pdf")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure 9: head ablation heatmap.
# -----------------------------------------------------------------------------
def fig9_head_ablation(p: int = 113) -> None:
    name = "05_multitask_three"
    if not _has_run(name):
        return
    model = _load_model(name, p=p)
    data = make_multitask(("+", "-", "*"), p=p, train_frac=0.30, seed=42)
    results = per_task_head_ablation(
        model, data.X_test, data.y_test, data.task_test, data.ops
    )
    H = model.cfg.n_heads
    ops = data.ops
    matrix = np.zeros((H, len(ops)))
    for j, op in enumerate(ops):
        if op not in results:
            continue
        for h, r in enumerate(results[op]):
            matrix[h, j] = r.loss_increase

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(ops)))
    ax.set_xticklabels([OPS_DISPLAY[op] for op in ops])
    ax.set_yticks(range(H))
    ax.set_yticklabels([f"head {i}" for i in range(H)])
    for i in range(H):
        for j in range(len(ops)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    color="white" if matrix[i, j] < matrix.max() * 0.6 else "black",
                    fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("loss increase under ablation")
    ax.set_title("per-head per-task ablation, multitask-3")
    fig.savefig(FIG_DIR / "fig9_head_ablation.pdf")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Tables.
# -----------------------------------------------------------------------------
def _grok_step(name: str, op: str | None = None, threshold: float = 0.95) -> int | None:
    if not _has_run(name):
        return None
    h = load_history(_run_dir(name))
    if op is None:
        return find_grok_step(h["step"].tolist(), h["test_acc"].tolist())
    key = f"test_acc_{op}"
    if key not in h:
        return None
    return find_grok_step(h["step"].tolist(), h[key].tolist())


def tab1_grok_steps_single() -> None:
    rows = [
        ("addition (\\textsc{add})", _grok_step("01_baseline_addition")),
        ("subtraction (\\textsc{sub})", _grok_step("02_baseline_subtraction")),
        ("multiplication (\\textsc{mul})", _grok_step("03_baseline_multiplication")),
    ]
    out = (
        "\\begin{tabular}{lr}\n"
        "\\toprule\n"
        "task & grok step \\\\\n"
        "\\midrule\n"
    )
    for label, step in rows:
        cell = "--" if step is None else f"{step:,}"
        out += f"{label} & {cell} \\\\\n"
    out += "\\bottomrule\n\\end{tabular}\n"
    (FIG_DIR / "tab1_grok_steps_single.tex").write_text(out)


def tab2_grok_steps_multi() -> None:
    runs = [
        ("seed 42", "05_multitask_three"),
        ("seed 137", "07_multitask_three_seed_137"),
        ("seed 271", "07_multitask_three_seed_271"),
    ]
    out = (
        "\\begin{tabular}{lrrr}\n"
        "\\toprule\n"
        " & \\textsc{add} & \\textsc{sub} & \\textsc{mul} \\\\\n"
        "\\midrule\n"
    )
    for label, name in runs:
        cells = []
        for op in ("+", "-", "*"):
            s = _grok_step(name, op)
            cells.append("--" if s is None else f"{s:,}")
        out += f"{label} & " + " & ".join(cells) + " \\\\\n"
    out += "\\bottomrule\n\\end{tabular}\n"
    (FIG_DIR / "tab2_grok_steps_multi.tex").write_text(out)


# -----------------------------------------------------------------------------
# Summary JSON for the paper / commit messages.
# -----------------------------------------------------------------------------
def write_summary() -> None:
    summary: dict = {}
    for name in [d.name for d in RUNS.iterdir() if d.is_dir()]:
        if not _has_run(name):
            continue
        h = load_history(_run_dir(name))
        entry = {
            "final_train_acc": float(h["train_acc"][-1]),
            "final_test_acc": float(h["test_acc"][-1]),
            "final_train_loss": float(h["train_loss"][-1]),
            "final_test_loss": float(h["test_loss"][-1]),
            "n_steps_logged": int(h["step"].max()),
            "grok_step_overall": _grok_step(name),
        }
        per_task = {}
        for op in ("+", "-", "*"):
            if f"test_acc_{op}" in h:
                per_task[op] = {
                    "final_test_acc": float(h[f"test_acc_{op}"][-1]),
                    "final_test_loss": float(h[f"test_loss_{op}"][-1]),
                    "grok_step": _grok_step(name, op),
                }
        if per_task:
            entry["per_task"] = per_task
        summary[name] = entry
    (RUNS / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


# -----------------------------------------------------------------------------
# Entry point.
# -----------------------------------------------------------------------------
def main() -> None:
    fig1_replication()
    fig2_per_op_baselines()
    fig3_multitask_two()
    fig4_multitask_three()
    fig5_seed_sweep()
    fig6_fourier_singletask()
    fig7_fourier_mult()
    fig8_feature_overlap()
    fig9_head_ablation()
    tab1_grok_steps_single()
    tab2_grok_steps_multi()
    write_summary()
    print(f"\nAll figures and tables written to {FIG_DIR}")


if __name__ == "__main__":
    main()
