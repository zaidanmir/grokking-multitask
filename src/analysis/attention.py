"""Attention-pattern visualisation for the 1-layer transformer.

After grokking, attention from the final ``=`` position to the operand tokens
is typically near-uniform (~50% on each operand) for the addition task,
indicating the model has discovered a position-symmetric representation.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def attention_patterns(
    model, X: torch.Tensor
) -> torch.Tensor:
    """Return per-head attention probabilities for the final position.

    Output shape: ``(B, H, T)`` — for each example and head, the attention
    distribution from the last token (``=``) over all four input tokens.
    """
    model.eval()
    cfg = model.cfg
    H, Dh = cfg.n_heads, cfg.d_head
    embeds = model.embed(X)
    qkv = model.attn.W_QKV(embeds)
    q, k, _ = qkv.chunk(3, dim=-1)
    B, T, _ = embeds.shape
    q = q.view(B, T, H, Dh).transpose(1, 2)         # (B, H, T, Dh)
    k = k.view(B, T, H, Dh).transpose(1, 2)
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(Dh)
    mask = torch.triu(
        torch.ones(T, T, dtype=torch.bool, device=embeds.device), 1
    )
    scores = scores.masked_fill(mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    # Take the row corresponding to the last (read-out) position.
    return attn[:, :, -1, :]
