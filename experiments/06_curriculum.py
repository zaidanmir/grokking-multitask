"""Curriculum experiment: addition first, then add subtract, then add multiply.

Train on a single task to grokking, save the checkpoint, then resume training
on the joint multi-task dataset. Compare time-to-grok per task against the
joint-from-start baseline (experiment 05).
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from data.tasks import make_multitask, make_task
from src.model import ModelConfig, OneLayerTransformer
from src.train import TrainConfig, train


def main() -> None:
    seed = 42
    p = 113
    train_frac = 0.30

    # Stage 1: addition only.
    add_data = make_task("+", p=p, train_frac=train_frac, seed=seed)
    cfg_a = TrainConfig(
        name="06_curriculum_stage_a_add",
        ops=("+",),
        p=p,
        train_frac=train_frac,
        seed=seed,
        steps=10_000,
        log_every=2000,
        eval_every=200,
    )
    out_a = train(cfg_a, add_data)
    state_a = out_a["model"].state_dict()

    # Stage 2: add the second task and continue. Re-init the model with the
    # stage-1 weights — this is the "curriculum" condition.
    multitask_data = make_multitask(("+", "-", "*"), p=p, train_frac=train_frac, seed=seed)
    cfg_b = TrainConfig(
        name="06_curriculum_stage_b_multi",
        ops=("+", "-", "*"),
        p=p,
        train_frac=train_frac,
        seed=seed,
        steps=65_000,
        log_every=2000,
        eval_every=500,
    )
    # Train, then patch the model's initial weights post-hoc by manually
    # constructing the model and stitching weights in before optimisation.
    # Re-implementing here keeps the patch tight without modifying train().
    device = cfg_b.resolve_device()
    model_cfg = ModelConfig(p=p)
    torch.manual_seed(seed)
    model = OneLayerTransformer(model_cfg).to(device)
    model.load_state_dict(state_a)
    multitask_data = multitask_data.to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=cfg_b.lr,
        betas=(cfg_b.beta1, cfg_b.beta2),
        eps=cfg_b.eps,
        weight_decay=cfg_b.weight_decay,
    )

    import time
    import json
    import numpy as np
    import torch.nn.functional as F
    from src.eval import evaluate
    from src.train import _save_history

    save_root = Path(cfg_b.save_dir) / cfg_b.name
    save_root.mkdir(parents=True, exist_ok=True)
    with (save_root / "config.yaml").open("w") as f:
        yaml.safe_dump(
            {
                "train": asdict(cfg_b),
                "model": asdict(model_cfg),
                "data": {
                    "ops": list(multitask_data.ops),
                    "p": p,
                    "train_frac": train_frac,
                    "seed": seed,
                    "warm_start_from": cfg_a.name,
                },
            },
            f,
            sort_keys=False,
        )

    history = []
    t0 = time.time()
    for step in range(cfg_b.steps + 1):
        if step > 0:
            model.train()
            logits = model(multitask_data.X_train)
            loss = F.cross_entropy(logits, multitask_data.y_train)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
        if step % cfg_b.eval_every == 0 or step == cfg_b.steps:
            tr = evaluate(
                model, multitask_data.X_train, multitask_data.y_train,
                multitask_data.task_train, multitask_data.ops,
            )
            te = evaluate(
                model, multitask_data.X_test, multitask_data.y_test,
                multitask_data.task_test, multitask_data.ops,
            )
            history.append(
                {
                    "step": step,
                    "train_loss": tr.loss,
                    "train_acc": tr.acc,
                    "test_loss": te.loss,
                    "test_acc": te.acc,
                    "per_task_train_loss": tr.per_task_loss,
                    "per_task_train_acc": tr.per_task_acc,
                    "per_task_test_loss": te.per_task_loss,
                    "per_task_test_acc": te.per_task_acc,
                }
            )
            if step % cfg_b.log_every == 0 or step == cfg_b.steps:
                parts = [
                    f"{op}={te.per_task_acc.get(op, float('nan')):.3f}"
                    for op in multitask_data.ops
                ]
                print(
                    f"[{cfg_b.name}] step {step:>6d}  "
                    f"train {tr.loss:.3e}/{tr.acc:.3f}  "
                    f"test {te.loss:.3e}/{te.acc:.3f}  "
                    f"per-task: " + " ".join(parts),
                    flush=True,
                )
    torch.save(model.state_dict(), save_root / "final.pt")
    _save_history(save_root, history, multitask_data.ops)
    print(f"[{cfg_b.name}] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
