"""Per-head and per-component ablation utilities.

The ``OneLayerTransformer`` in :py:mod:`src.model` exposes per-head residual
contributions through ``forward_with_internals``. The ablation utilities
zero out a chosen head's contribution and re-evaluate the loss, returning
the change relative to the unablated baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class AblationResult:
    head: int
    loss_before: float
    loss_after: float
    acc_before: float
    acc_after: float

    @property
    def loss_increase(self) -> float:
        return self.loss_after - self.loss_before


@torch.no_grad()
def _baseline(model, X, y) -> tuple[float, float]:
    model.eval()
    logits = model(X)
    loss = F.cross_entropy(logits, y).item()
    acc = (logits.argmax(-1) == y).float().mean().item()
    return loss, acc


@torch.no_grad()
def ablate_head(
    model, head_idx: int, X: torch.Tensor, y: torch.Tensor
) -> AblationResult:
    """Zero out the contribution of one attention head and measure loss.

    Implementation: re-run the forward pass, but reconstruct the post-attention
    activation as ``embeds + sum(per_head[h] for h != head_idx)``. The MLP is
    then run as normal.
    """
    model.eval()
    loss_before, acc_before = _baseline(model, X, y)

    internals = model.forward_with_internals(X)
    embeds = internals["embeds"]
    per_head = internals["attn_per_head"]              # (H, B, T, D)
    H = per_head.shape[0]
    if not 0 <= head_idx < H:
        raise IndexError(f"head_idx {head_idx} out of range [0, {H})")
    mask = torch.ones(H, device=per_head.device)
    mask[head_idx] = 0.0
    attn_kept = (per_head * mask.view(H, 1, 1, 1)).sum(dim=0)
    x_post_attn = embeds + attn_kept
    mlp_out = model.mlp(x_post_attn)
    x_final = x_post_attn + mlp_out
    logits = model.W_U(x_final[:, -1, :])
    loss_after = F.cross_entropy(logits, y).item()
    acc_after = (logits.argmax(-1) == y).float().mean().item()
    return AblationResult(
        head=head_idx,
        loss_before=loss_before,
        loss_after=loss_after,
        acc_before=acc_before,
        acc_after=acc_after,
    )


def ablate_all_heads(
    model, X: torch.Tensor, y: torch.Tensor
) -> list[AblationResult]:
    H = model.cfg.n_heads
    return [ablate_head(model, h, X, y) for h in range(H)]


@torch.no_grad()
def per_task_head_ablation(
    model,
    X: torch.Tensor,
    y: torch.Tensor,
    task_ids: torch.Tensor,
    ops: tuple[str, ...],
) -> dict[str, list[AblationResult]]:
    """Run :py:func:`ablate_head` separately on each task's slice of the data."""
    out: dict[str, list[AblationResult]] = {}
    for idx, op in enumerate(ops):
        mask = task_ids == idx
        if not mask.any():
            continue
        out[op] = ablate_all_heads(model, X[mask], y[mask])
    return out
