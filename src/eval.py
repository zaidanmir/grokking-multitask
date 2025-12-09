"""Evaluation utilities: per-task loss and accuracy, with optional sub-batching."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class EvalResult:
    loss: float
    acc: float
    per_task_loss: dict[str, float]
    per_task_acc: dict[str, float]


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    task_ids: torch.Tensor,
    ops: tuple[str, ...],
    batch_size: int | None = None,
) -> EvalResult:
    """Compute mean cross-entropy and top-1 accuracy, sliced by task.

    The full ``X`` typically fits in memory at this scale; ``batch_size`` is
    provided for the case of very large ``p`` or many tasks.
    """
    model.eval()
    n = X.shape[0]
    if batch_size is None or batch_size >= n:
        logits = model(X)
        loss_per = F.cross_entropy(logits, y, reduction="none")
        pred = logits.argmax(dim=-1)
    else:
        loss_chunks, pred_chunks = [], []
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            logits = model(X[i:j])
            loss_chunks.append(F.cross_entropy(logits, y[i:j], reduction="none"))
            pred_chunks.append(logits.argmax(dim=-1))
        loss_per = torch.cat(loss_chunks)
        pred = torch.cat(pred_chunks)

    correct = (pred == y).float()
    overall_loss = loss_per.mean().item()
    overall_acc = correct.mean().item()

    per_task_loss: dict[str, float] = {}
    per_task_acc: dict[str, float] = {}
    for idx, op in enumerate(ops):
        mask = task_ids == idx
        if mask.any():
            per_task_loss[op] = loss_per[mask].mean().item()
            per_task_acc[op] = correct[mask].mean().item()
    return EvalResult(
        loss=overall_loss,
        acc=overall_acc,
        per_task_loss=per_task_loss,
        per_task_acc=per_task_acc,
    )


def find_grok_step(
    steps: list[int], test_acc: list[float], threshold: float = 0.95
) -> int | None:
    """Return the first step at which test accuracy exceeds ``threshold``.

    ``None`` if it never does. The threshold of 0.95 follows Nanda 2023.
    """
    for s, a in zip(steps, test_acc):
        if a >= threshold:
            return s
    return None
