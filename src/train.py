"""Training loop for the grokking experiments.

Full-batch AdamW with the published Nanda 2023 hyperparameters. The loop
logs train and test (per-task) metrics to a list-of-dicts and persists them
on disk as both NPZ (machine-readable) and YAML (human-readable). The full
config of every run is saved alongside so the run is reproducible from disk
alone.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import json
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from data.tasks import TaskData
from src.eval import evaluate, EvalResult
from src.model import ModelConfig, OneLayerTransformer


@dataclass
class TrainConfig:
    name: str
    ops: tuple[str, ...]
    p: int = 113
    train_frac: float = 0.30
    seed: int = 42
    steps: int = 30_000
    lr: float = 1e-3
    weight_decay: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1e-8
    log_every: int = 100
    eval_every: int = 100
    checkpoint_every: int = 0          # 0 = never (only final)
    device: str = "auto"
    save_dir: str = "runs"

    def resolve_device(self) -> torch.device:
        if self.device != "auto":
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


def train(
    cfg: TrainConfig, data: TaskData, model_cfg: Optional[ModelConfig] = None
) -> dict:
    """Run one training experiment end-to-end.

    Returns a dict containing the trained model, the run config, and the
    metric history. Side-effect: writes config + metrics to ``cfg.save_dir/cfg.name``.
    """
    device = cfg.resolve_device()
    model_cfg = model_cfg or ModelConfig(p=cfg.p)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    model = OneLayerTransformer(model_cfg).to(device)
    data = data.to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )

    history: list[dict] = []
    save_root = Path(cfg.save_dir) / cfg.name
    save_root.mkdir(parents=True, exist_ok=True)
    ckpt_root = save_root / "checkpoints"
    if cfg.checkpoint_every > 0:
        ckpt_root.mkdir(exist_ok=True)

    # Persist config up-front.
    with (save_root / "config.yaml").open("w") as f:
        yaml.safe_dump(
            {
                "train": asdict(cfg),
                "model": asdict(model_cfg),
                "data": {
                    "ops": list(data.ops),
                    "p": data.p,
                    "train_frac": data.train_frac,
                    "seed": data.seed,
                    "n_train": int(data.X_train.shape[0]),
                    "n_test": int(data.X_test.shape[0]),
                },
            },
            f,
            sort_keys=False,
        )

    print(
        f"[{cfg.name}] device={device} train_n={data.X_train.shape[0]} "
        f"test_n={data.X_test.shape[0]} ops={data.ops} steps={cfg.steps}"
    )
    t0 = time.time()
    for step in range(cfg.steps + 1):
        if step > 0:
            model.train()
            logits = model(data.X_train)
            loss = F.cross_entropy(logits, data.y_train)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()

        if step % cfg.eval_every == 0 or step == cfg.steps:
            train_eval = evaluate(
                model, data.X_train, data.y_train, data.task_train, data.ops
            )
            test_eval = evaluate(
                model, data.X_test, data.y_test, data.task_test, data.ops
            )
            history.append(
                {
                    "step": step,
                    "train_loss": train_eval.loss,
                    "train_acc": train_eval.acc,
                    "test_loss": test_eval.loss,
                    "test_acc": test_eval.acc,
                    "per_task_train_loss": train_eval.per_task_loss,
                    "per_task_train_acc": train_eval.per_task_acc,
                    "per_task_test_loss": test_eval.per_task_loss,
                    "per_task_test_acc": test_eval.per_task_acc,
                }
            )
            if step % (cfg.log_every) == 0 or step == cfg.steps:
                msg = (
                    f"[{cfg.name}] step {step:>6d}  "
                    f"train {train_eval.loss:.3e}/{train_eval.acc:.3f}  "
                    f"test {test_eval.loss:.3e}/{test_eval.acc:.3f}"
                )
                if len(data.ops) > 1:
                    parts = [
                        f"{op}={test_eval.per_task_acc.get(op, float('nan')):.3f}"
                        for op in data.ops
                    ]
                    msg += "  per-task test acc: " + " ".join(parts)
                print(msg, flush=True)

        if cfg.checkpoint_every > 0 and step > 0 and step % cfg.checkpoint_every == 0:
            torch.save(model.state_dict(), ckpt_root / f"step_{step:06d}.pt")

    # Final save: weights + history.
    torch.save(model.state_dict(), save_root / "final.pt")
    _save_history(save_root, history, data.ops)
    elapsed = time.time() - t0
    print(f"[{cfg.name}] done in {elapsed:.1f}s, saved to {save_root}")
    return {"model": model, "history": history, "save_dir": save_root}


def _save_history(save_root: Path, history: list[dict], ops: tuple[str, ...]) -> None:
    """Persist metrics as both an NPZ (for plotting) and JSONL (for browsing)."""
    steps = np.array([h["step"] for h in history])
    arrays = {
        "step": steps,
        "train_loss": np.array([h["train_loss"] for h in history]),
        "train_acc": np.array([h["train_acc"] for h in history]),
        "test_loss": np.array([h["test_loss"] for h in history]),
        "test_acc": np.array([h["test_acc"] for h in history]),
    }
    for op in ops:
        arrays[f"train_loss_{op}"] = np.array(
            [h["per_task_train_loss"].get(op, np.nan) for h in history]
        )
        arrays[f"train_acc_{op}"] = np.array(
            [h["per_task_train_acc"].get(op, np.nan) for h in history]
        )
        arrays[f"test_loss_{op}"] = np.array(
            [h["per_task_test_loss"].get(op, np.nan) for h in history]
        )
        arrays[f"test_acc_{op}"] = np.array(
            [h["per_task_test_acc"].get(op, np.nan) for h in history]
        )
    np.savez(save_root / "history.npz", **arrays)
    with (save_root / "history.jsonl").open("w") as f:
        for row in history:
            f.write(json.dumps(row) + "\n")


def load_history(run_dir: Path | str) -> dict[str, np.ndarray]:
    """Reload an NPZ history written by ``train``."""
    npz = np.load(Path(run_dir) / "history.npz")
    return {k: npz[k] for k in npz.files}
